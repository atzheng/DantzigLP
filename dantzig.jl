using JuMP, Gurobi, MathProgBase, Lasso, DataFrames

TOL = 1e-8

"""
Struct to hold constraint refs and mutable variables for column
and constraint generation.
"""
abstract type DantzigModel end

type BasicDantzigModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    pos_betas        # Postive Beta variables already in the model (mutable)
    pos_beta_indices # Indices corresponding to entries of pos_betas (mutable)
    neg_betas        # Negative Beta variables already in the model (mutable)
    neg_beta_indices # Indices corresponding to entries of neg_betas (mutable)
    residuals        # Residual variables
    residual_constrs # Constraints r = y - X * Beta
    linf_pos_constrs # Constraints X'r <= delta
    linf_neg_constrs # Constraints X'r >= -delta
    X                # Data matrix X
    # Diagnostics
    diagnostics      # DataFrame of diagnostics
    initializer_secs # Initializer runtime
end

Base.show(io::IO, x::DantzigModel) = show(io, x.gurobi_model)


"""
Baseline solution. Mainly for testing.
Under normal conditions, should be identical to dantzig_lp with
column / constraint generation disabled.
"""
function baseline_dantzig(y, X, delta, verbose = false)
    n, p = size(X)

    model = Model(solver = GurobiSolver(OutputFlag = ifelse(verbose, 1, 0)))

    residuals = @variable(model, [1:n])
    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)

    residual_constrs =
        @constraint(model, y - X * (abs_beta_pos - abs_beta_neg) .== residuals)
    linf_pos_constrs = @constraint(model, X' * residuals .<= delta)
    linf_neg_constrs = @constraint(model, X' * residuals .>= -delta)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    if verbose info("Solving model...") end
    solve(model)
    return model, sparse(getvalue(abs_beta_pos) - getvalue(abs_beta_neg))
end


function dantzig_lp(y, X, delta;
                    initializer_fn = lasso_initializer,
                    colgen_fn = generate_columns!,
                    colgen_args = [(:max_columns, 40)],
                    congen_fn = generate_constraints!,
                    congen_args = [(:max_constraints, 50)],
                    constraint_generation = true,
                    column_generation = true,
                    verbose = false,
                    timeout = Inf,
                    model_initializer = initialize_model)

    model = model_initializer(
        X, y, minimum(delta), initializer_fn,
        constraint_generation, column_generation, verbose)

    partial_colgen_fn(model) = colgen_fn(model; colgen_args...)
    partial_congen_fn(model, delta) = congen_fn(model, delta; congen_args...)
    solver(model, delta) = solve_model(model, delta, partial_colgen_fn,
                                       constraint_generation, partial_congen_fn,
                                       column_generation, verbose, timeout)

    if isa(delta, Array)  # Solve a path of deltas
        solutions = []
        for d in delta  # TODO enforce an ordering of deltas
            for constr in model.linf_pos_constrs
                JuMP.setRHS(constr, d)
            end
            for constr in model.linf_neg_constrs
                JuMP.setRHS(constr, -d)
            end
            solution = solver(model, d)
            push!(solutions, solution)
        end
        diagnostics = vcat([soln[2] for soln in solutions]...)
        coefs = [soln[1] for soln in solutions]
    else
        coefs, diagnostics = solver(model, delta)
    end
    model.diagnostics = diagnostics
    return (model, coefs)
end


function solve_model(model, delta, colgen_fn,
                     constraint_generation, congen_fn,
                     column_generation, verbose, timeout)
    # Intialize tracking for diagnostics
    start_time = time_ns()
    columns_generated = 0
    column_generation_seconds = 0
    constraint_generation_seconds = 0
    constraints_generated = 0
    gurobi_seconds = 0
    initial_vars = union(model.pos_beta_indices, model.neg_beta_indices)
    solve_status, solve_time = @timed solve(model.gurobi_model)

    if solve_status == :Infeasible
        warn("Infeasible initial model.")
        return solve_status, DataFrame()
    end

    gurobi_seconds += solve_time
    status = :InProgress

    # Start solver
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        if (time_ns() - start_time) / 1.0e9 > timeout
            status = :Timeout
            break
        end

        # Constraint generation
        if constraint_generation
            congen_start_ts = time_ns()
            pos_constrs = []
            neg_constrs = []

            new_pos_constrs, new_neg_constrs = congen_fn(model, delta)
            push!(pos_constrs, new_pos_constrs)
            push!(neg_constrs, new_neg_constrs)

            while length(new_pos_constrs) + length(new_neg_constrs) > 0
                solve_status, solve_time = @timed solve(model.gurobi_model)
                gurobi_seconds += solve_time
                new_pos_constrs, new_neg_constrs = congen_fn(model, delta)
                push!(pos_constrs, new_pos_constrs)
                push!(neg_constrs, new_neg_constrs)
            end

            flat_pos_constrs = vcat(pos_constrs...)
            flat_neg_constrs = vcat(neg_constrs...)

            n_new_constrs = length(flat_pos_constrs) + length(flat_neg_constrs)
            constraints_generated += n_new_constrs
            if verbose info(@sprintf("Generated %d constraints.",
                                     n_new_constrs)) end

            constraint_generation_seconds +=
                (time_ns() - congen_start_ts) / 1.0e9
        end

        # Column generation
        if column_generation
            colgen_start_ts = time_ns()
            new_columns = colgen_fn(model)
            if verbose info(@sprintf("Generated %d columns.",
                                    length(new_constrs))) end
            if length(new_columns) == 0
                status = :Optimal
            else
                solve_status, solve_time = @timed solve(model.gurobi_model)
                gurobi_seconds += solve_time
                columns_generated += length(new_columns)
            end
            column_generation_seconds += (time_ns() - colgen_start_ts) / 1.0e9
        else
            status = :Optimal
        end
    end

    # Extract solution
    beta_values = spzeros(size(model.X)[2])
    for (var, i) in zip(model.pos_betas, model.pos_beta_indices)
        beta_values[i] = getvalue(var)
    end

    for (var, i) in zip(model.neg_betas, model.neg_beta_indices)
        beta_values[i] -= getvalue(var)
    end

    # Tally up diagnostics
    end_time = time_ns()
    total_seconds = (end_time - start_time) / 1.0e9
    correct_initial_vars = length(intersect(beta_values.nzind, initial_vars))
    total_vars = length(beta_values.nzind)
    diagnostics = DataFrame(
        delta = delta,
        total_seconds = total_seconds,
        gurobi_seconds = gurobi_seconds,
        columns_generated = columns_generated,
        column_generation_seconds = column_generation_seconds,
        constraints_generated = constraints_generated,
        constraint_generation_seconds = constraint_generation_seconds,
        initial_vars = length(initial_vars),
        correct_initial_vars = correct_initial_vars,
        total_vars = total_vars,
        status = string(status))
    return beta_values, diagnostics
end


# Column / Constraint Generation
# ==============================================================================
# Column Generation
# ------------------------------------------------------------------------------
"""
Closure to add new beta variables to the model.
WARNING: mutates betas and beta_indices.
"""
function add_beta!(model, idx, sgn)
    if sgn > 0
        beta_indices = model.pos_beta_indices
        betas = model.pos_betas
    else
        beta_indices = model.neg_beta_indices
        betas = model.neg_betas
    end

    if idx in beta_indices
        return nothing
    else
        new_var = @variable(
            model.gurobi_model,
            objective = 1,
            inconstraints = model.residual_constrs,
            coefficients = -sgn * full(model.X[:, idx]),
            lowerbound = 0)
        push!(beta_indices, idx)
        push!(betas, new_var)
        return new_var
    end
end


function generate_columns!(model;
                           max_columns = 5,
                           beta_adder = add_beta!,
                           reduced_cost_fn = get_reduced_costs)
    # Compute reduced costs
    pos_costs, neg_costs = reduced_cost_fn(model)
    pos_costs[model.pos_beta_indices] = Inf
    neg_costs[model.neg_beta_indices] = Inf

    max_columns = min(max_columns, length(pos_costs))
    pos_min_indices = sortperm(vec(pos_costs))[1:max_columns]
    neg_min_indices = sortperm(vec(neg_costs))[1:max_columns]

    # Sort reduced cost vectors
    # (idx, sgn) tuples for return values
    pos_return_vals = zip_collect(
        pos_min_indices, ones(length(pos_min_indices)))
    neg_return_vals = zip_collect(
        neg_min_indices, -ones(length(neg_min_indices)))

    pos_min_costs = zip_collect(pos_costs[pos_min_indices], pos_return_vals)
    neg_min_costs = zip_collect(neg_costs[neg_min_indices], neg_return_vals)

    all_costs = vcat(pos_min_costs, neg_min_costs)
    new_columns = [x[2] for x in all_costs if x[1] < -TOL]

    # Add betas
    for (idx, sign) in new_columns
        beta_adder(model, idx, sign)
    end

    return new_columns
end

zip_collect(args...) = collect(zip(map(vec, args)...))

function get_reduced_costs(model)
    n, p = size(model.X)

    # Compute reduced costs separately for the positive and negative
    # components of Beta
    duals = model.gurobi_model.linconstrDuals[1:n]
    pos_reduced_costs = 1 + xtA(duals, model.X)
    neg_reduced_costs = 1 - xtA(duals, model.X)
    return (pos_reduced_costs, neg_reduced_costs)
end


xtA(x::Vector, A::Matrix) = x'A


# Constraint Generation
# ------------------------------------------------------------------------------
function generate_constraints!(model, delta;
                               max_constraints = 50,
                               verbose = false,
                               constr_adder = add_Xtr_constr!)
    X = model.X
    n, p = size(X)
    TOL = 1e-6
    resid_vals = getvalue(model.residuals)
    constraint_values = vec(xtA(resid_vals, X))
    constraint_indices = sortperm(abs.(constraint_values), rev = true)

    new_pos_constrs = []
    new_neg_constrs = []
    for row in constraint_indices[1:min(max_constraints, p)]
        val = constraint_values[row]
        if val < -delta - TOL
            new_constr = constr_adder(model, delta, row, -1)
            push!(new_neg_constrs, new_constr)
        elseif val > delta + TOL
            new_constr = constr_adder(model, delta, row, 1)
            push!(new_pos_constrs, new_constr)
        end
    end
    return new_pos_constrs, new_neg_constrs
end


function add_Xtr_constr!(model, delta, idx, sign)
    if sign == -1
        new_constr = @constraint(
            model.gurobi_model,
            dot(model.X[:, idx], model.residuals) >= -delta)
        push!(model.linf_neg_constrs, new_constr)
    elseif sign == 1
        new_constr = @constraint(
            model.gurobi_model,
            dot(model.X[:, idx], model.residuals) <= delta)
        push!(model.linf_pos_constrs, new_constr)
    end
    return new_constr
end


# Initialization Functions
# ==============================================================================
function initialize_model(X, y, delta, initializer_fn,
                          constraint_generation, column_generation, verbose)
    # --- Construct model ---
    n, p = size(X)

    output_flag = if verbose 1 else 0 end
    solver = GurobiSolver(Method = -1, OutputFlag = output_flag)
    gurobi_model = Model(solver = solver)

    residuals = @variable(gurobi_model, [1:n])
    residual_constrs = @constraint(gurobi_model, y - residuals .== 0)
    obj = @objective(gurobi_model, Min, 0)

    # --- Generate Lasso Solution ---
    if verbose info("Finding initial solution...") end
    initializer_soln, initializer_seconds = @timed initializer_fn(X, y, delta)
    if verbose info(@sprintf("Initial solution found in %d seconds",
                             initializer_seconds)) end
    # --- Initialize constraints ---
    if constraint_generation == false
        init_constrs = 1:p
    else
        # initialize with constraints from Lasso support
        init_constrs = initializer_soln.nzind
    end


    # TODO Only need to add either the positive or negative constraints
    linf_pos_constrs =
        @constraint(gurobi_model, X[:, init_constrs]' * residuals .<= delta)
    linf_neg_constrs =
        @constraint(gurobi_model, X[:, init_constrs]' * residuals .>= -delta)

    model = BasicDantzigModel(gurobi_model, [], [], [], [],
                              residuals, residual_constrs,
                              linf_pos_constrs, linf_neg_constrs, X,
                              nothing, initializer_seconds)

    # --- Initialize variables
    if column_generation
        for idx in initializer_soln.nzind
            coef = initializer_soln[idx]
            # Only need to add one sign, but that doesn't work with an array
            # of deltas. TODO
            new_var = add_beta!(model, idx, sign(coef))
            new_var = add_beta!(model, idx, -sign(coef))
            setvalue(new_var, coef)
        end
    else
        for idx in 1:p
            add_beta!(model, idx, 1)
            add_beta!(model, idx, -1)
        end
    end
    return model
end


function lasso_initializer(X, y, delta::Number)
    return lasso_initializer(X, y, [delta])
end


function lasso_initializer(X, y, delta::Array)
    n, p = size(X)
    soln = fit(LassoPath, X, y, λ = delta ./ n,
               maxncoef = p, standardize = false).coefs
    return merge_vectors([soln[:, j] for j in 1:size(soln, 2)]...)
end


function max_correlation_initializer(X, y, delta;
                                     coefs = Int(round(0.2 * length(y))))
    n, p = size(X)
    Xty = X'y
    nzind = sortperm(abs.(Xty), rev = true)[1:coefs]
    results = spzeros(p)
    results[nzind] = Xty[nzind]
    return results
end


function composite_initializer(X, y, delta, initializer_fns)
    initial_indices = [initializer_fn(X, y, delta)
                       for initializer_fn in initializer_fns]
    return merge_vectors(initial_indices...)
end


function recursive_initializer(X, y, delta, initializer_fn, max_depth)
    soln = initializer_fn(X, y, delta)
    if max_depth == 1
        return soln
    else
        p = size(X, 2)
        next_indices = sort(setdiff(1:p, soln.nzind))
        next_vector = spzeros(p)
        next_results = recursive_initializer(
            X[:, next_indices], y, delta, initializer_fn, max_depth - 1)
        next_vector[next_indices] = next_results
        return merge_vectors(soln, next_vector)
    end
end


"""
Merge the nonzero values of all vectors, with smaller indices of xs
taking precedence.
"""
function merge_vectors(xs...)
    if length(unique(map(length, xs))) > 1
        error("All vectors must have equal length.")
    end

    results = spzeros(length(xs[1]))
    nzinds = []
    for x in xs
        spx = sparse(x)
        new_nzind = setdiff(spx.nzind, nzinds)
        results[new_nzind] = spx[new_nzind]
        push!(nzinds, new_nzind)
    end

    return results
end

# Module end
# end
