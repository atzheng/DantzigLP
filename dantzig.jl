using JuMP, Gurobi, MathProgBase, Lasso, DataFrames

TOL = 1e-15

"""
Struct to hold constraint refs and mutable variables for column
and constraint generation.
"""
abstract type DantzigModel end
Base.show(io::IO, x::DantzigModel) = show(io, x.gurobi_model)

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
end

function BasicDantzigModel(X, y)
    n, p = size(X)
    gurobi_model = Model()

    residuals = @variable(gurobi_model, [1:n])
    residual_constrs = @constraint(gurobi_model, y - residuals .== 0)
    obj = @objective(gurobi_model, Min, 0)

    model = BasicDantzigModel(gurobi_model, [], [], [], [],
                              residuals, residual_constrs,
                              [], [], X, nothing)
    return model
end


"""
Baseline solution. Mainly for testing.
Under normal conditions, should be identical to dantzig_lp with
column / constraint generation disabled.
"""
function baseline_dantzig(X, y, λ, verbose = false)
    n, p = size(X)

    model = Model(solver = GurobiSolver(OutputFlag = ifelse(verbose, 1, 0)))

    residuals = @variable(model, [1:n])
    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)

    residual_constrs =
        @constraint(model, y - X * (abs_beta_pos - abs_beta_neg) .== residuals)
    linf_pos_constrs = @constraint(model, X' * residuals .<= λ)
    linf_neg_constrs = @constraint(model, X' * residuals .>= -λ)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    if verbose info("Solving model...") end
    solve(model)
    return model, sparse(getvalue(abs_beta_pos) - getvalue(abs_beta_neg))
end


function dantzig_lp(X, y, λ; initializer_fn = lasso_initializer, args...)
    model = construct_basic_dantzig_model(X, y)
    initial_soln, initializer_seconds = @timed initializer_fn(X, y, λ)
    soln, log = solve_dantzig_lp!(model, λ, initial_soln; args...)
    return soln, model, log
end


function solve_dantzig_lp!(model, λ, initial_soln;
                           column_generation = true, max_columns = 40,
                           constraint_generation = true, max_constraints = 50,
                           verbose = false, timeout = Inf,
                           solver_params = Dict())

    start_time = time_ns()
    # Initialize
    # --------------------------------------------------------------------------
    default_solver_params = Dict([(:Method, 1),
                                  (:OutputFlag, ifelse(verbose, 1, 0)),
                                  (:TimeLimit, timeout)])
    solver_params_w_defaults = merge(default_solver_params, solver_params)
    solver = GurobiSolver(; solver_params_w_defaults...)
    setsolver(model.gurobi_model, solver)

    if column_generation
        initialize_columns!(model, initial_soln)
    else
        initialize_columns!(model)
    end

    if constraint_generation
        initialize_constraints!(model, minimum(λ), initial_soln)
    else
        initialize_constraints!(model, minimum(λ))
    end

    solver(modelx, λx) = solve_model(modelx, λx,
                                     column_generation, max_columns,
                                     constraint_generation, max_constraints,
                                     verbose, timeout)

    if verbose
        info(@sprintf("Model construction finished in %.2f seconds.",
                      (time_ns() - start_time) / 1e9))
    end

    # Solve
    # --------------------------------------------------------------------------
    if isa(λ, Array)  # Solve a path of deltas
        solutions = []
        for d in λ # TODO enforce an ordering of deltas
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
        coefs, diagnostics = solver(model, λ)
    end
    return (coefs, diagnostics)
end


function solve_model(model, delta,
                     column_generation, max_columns,
                     constraint_generation, max_constraints,
                     verbose, timeout)
    # Intialize tracking for diagnostics
    start_time = time_ns()
    columns_generated = 0
    column_generation_seconds = 0
    constraint_generation_seconds = 0
    constraints_generated = 0
    gurobi_seconds = 0
    initial_vars = union(model.pos_beta_indices, model.neg_beta_indices)
    solve_status, solve_time = @timed solve(model.gurobi_model)
    @show solve_status

    # if solve_status == :Infeasible
    #     warn("Infeasible initial model.")
    #     return solve_status, DataFrame()
    # end

    gurobi_seconds += solve_time
    status = :InProgress

    # iterations = []

    # Start solver
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        seconds_elapsed = (time_ns() - start_time) / 1.0e9
        if seconds_elapsed > timeout
            status = :Timeout
            break
        end

        # feas = norm(model.X'getvalue(model.residuals), Inf) - delta
        # pct_feas = (norm(model.X'getvalue(model.residuals), Inf) - delta) / delta
        # obj = (norm(getvalue(model.pos_betas), 1) +
        #        norm(getvalue(model.neg_betas), 1))
        # iteration_log = DataFrame(seconds = seconds_elapsed,
        #                           feas = feas,
        #                           pct_feas = pct_feas,
        #                           obj = obj)
        # push!(iterations, iteration_log)

        # Constraint generation
        if constraint_generation
            congen_start_ts = time_ns()
            pos_constrs = []
            neg_constrs = []

            new_pos_constrs, new_neg_constrs =
                generate_constraints!(model, delta)
            push!(pos_constrs, new_pos_constrs)
            push!(neg_constrs, new_neg_constrs)

            while length(new_pos_constrs) + length(new_neg_constrs) > 0
                solve_status, solve_time = @timed solve(model.gurobi_model)
                # if solve_status == :Infeasible
                #     break
                # end
                gurobi_seconds += solve_time
                new_pos_constrs, new_neg_constrs =
                    generate_constraints!(model, delta)
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
            new_columns = generate_columns!(model)
            if verbose info(@sprintf("Generated %d columns.",
                                    length(new_columns))) end
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
        # iterations = vcat(iterations...))

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
function add_beta!(model::BasicDantzigModel, idx::Integer, sgn::Number)
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


function add_beta!(model::DantzigModel, idx::Union{UnitRange, Vector},
                   sgn::Vector)
    for (i, s) in zip(idx, sgn)
        add_beta!(model, i, s)
    end
end


function generate_columns!(model::DantzigModel; max_columns=5)
    # Compute reduced costs
    pos_costs, neg_costs = get_reduced_costs(model)
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
        add_beta!(model, idx, Integer(sign))
    end

    return new_columns
end

zip_collect(args...) = collect(zip(map(vec, args)...))

function get_reduced_costs(model::BasicDantzigModel)
    n, p = size(model.X)

    # Compute reduced costs separately for the positive and negative
    # components of Beta
    # First n constraints correspond to the β constraints.
    # TODO Shouldn't need to make this assumption....
    duals = model.gurobi_model.linconstrDuals[1:n]
    pos_reduced_costs = 1 + xtA(duals, model.X)
    neg_reduced_costs = 1 - xtA(duals, model.X)
    return (pos_reduced_costs, neg_reduced_costs)
end


xtA(x::Vector, A::Matrix) = x'A


# Constraint Generation
# ------------------------------------------------------------------------------
function generate_constraints!(model::DantzigModel, delta;
                               max_constraints = 50)
    X = model.X
    n, p = size(X)
    TOL = 1e-15
    constraint_values = get_constraint_violations(model)
    constraint_indices = sortperm(abs.(constraint_values), rev = true)

    new_pos_constrs = []
    new_neg_constrs = []
    for row in constraint_indices[1:min(max_constraints, p)]
        val = constraint_values[row]
        if val < -delta - TOL
            new_constr = add_Xtr_constr!(model, delta, row, -1)
            push!(new_neg_constrs, new_constr)
        elseif val > delta + TOL
            new_constr = add_Xtr_constr!(model, delta, row, 1)
            push!(new_pos_constrs, new_constr)
        end
    end
    return new_pos_constrs, new_neg_constrs
end


function get_constraint_violations(model::BasicDantzigModel)
    resid_vals = getvalue(model.residuals)
    return vec(xtA(resid_vals, X))
end


function add_Xtr_constr!(model::BasicDantzigModel, delta::Number,
                         idx::Integer, sign::Integer)
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


function add_Xtr_constr!(model::DantzigModel, delta::Number,
                         idx::Union{UnitRange, Vector}, sgn::Vector)
    for (i, s) in zip(idx, sgn)
        add_Xtr_constr!(model, delta, i, s)
    end
end


# Initialization Functions
# ==============================================================================
function initialize_columns!(model)
    n, p = size(model.X)
    init_cols = 1:p
    add_beta!(model, init_cols, fill(1, p))
    add_beta!(model, init_cols, fill(-1, p))
end


function initialize_columns!(model, initial_soln)
    for idx in initial_soln.nzind
        coef = initial_soln[idx]
        # Only need to add one sign, but that doesn't work with an array
        # of deltas. TODO
        new_var = add_beta!(model, idx, Integer(sign(coef)))
        new_var = add_beta!(model, idx, Integer(-sign(coef)))
        setvalue(new_var, coef)
    end
end


function initialize_constraints!(model, λ)
    n, p = size(model.X)
    init_constrs = 1:p
    add_Xtr_constr!(model, λ, init_constrs, fill(1, p))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, p))
end


function initialize_constraints!(model, λ, initial_soln)
    init_constrs = initial_soln.nzind
    n_nz = length(initial_soln.nzind)
    add_Xtr_constr!(model, λ, init_constrs, fill(1, n_nz))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, n_nz))
end

#
# ==============================================================================
function lasso_initializer(X, y, delta::Number; args...)
    return lasso_initializer(X, y, [delta], args...)
end


function lasso_initializer(X, y, delta::Array; args...)
    n, p = size(X)
    soln = fit(LassoPath, X, y, λ = delta ./ n;
               maxncoef = p, standardize = false, cd_tol = 1e-15, args...,
               ).coefs
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
