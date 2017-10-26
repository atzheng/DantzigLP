module DantzigLP
using JuMP, Gurobi, MathProgBase, Lasso


"""Struct to hold diagnostic info from the Dantzig solver."""
struct dantzig_diagnostics
    total_seconds
    lasso_seconds
    gurobi_seconds
    columns_generated
    constraints_generated
    correct_lasso_vars
    total_vars
    status
end


"""
Struct to hold constraint refs and mutable variables for column
and constraint generation.
"""
type dantzig_model
    gurobi_model      # Gurobi model object (mutable)
    pos_betas         # Postive Beta variables already in the model (mutable)
    pos_beta_indices  # Indices corresponding to entries of pos_betas (mutable)
    neg_betas         # Negative Beta variables already in the model (mutable)
    neg_beta_indices  # Indices corresponding to entries of neg_betas (mutable)
    residuals         # Residual variables
    residual_constrs  # Constraints r = y - X * Beta
    linf_pos_constrs  # Constraints X'r <= delta
    linf_neg_constrs  # Constraints X'r >= -delta
    X                 # Data matrix X
end


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


function initialize_model(X, y, delta,
                          constraint_generation, column_generation, verbose)
    # --- Construct model ---
    n, p = size(X)
    I_p = speye(p)

    output_flag = if verbose 1 else 0 end
    solver = GurobiSolver(Method = 1, OutputFlag = output_flag)
    gurobi_model = Model(solver = solver)

    residuals = @variable(gurobi_model, [1:n])
    residual_constrs = @constraint(gurobi_model, y - residuals .== 0)
    obj = @objective(gurobi_model, Min, 0)

    # --- Generate Lasso Solution ---
    if verbose info("Fitting Lasso solution...") end
    lasso_soln, lasso_seconds =
        @timed vec(fit(LassoPath, X, y, λ = [delta / n],
                       maxncoef = p, standardize = false).coefs)

    # --- Initialize constraints ---
    if constraint_generation == false
        init_constrs = 1:p
    else
        # initialize with constraints from Lasso support
        init_constrs = lasso_soln.nzind
    end

    linf_pos_constrs =
        @constraint(gurobi_model, X'[init_constrs, :] * residuals .<= delta)
    linf_neg_constrs =
        @constraint(gurobi_model, X'[init_constrs, :] * residuals .>= -delta)

    model = dantzig_model(gurobi_model, [], [], [], [],
                          residuals, residual_constrs,
                          linf_pos_constrs, linf_neg_constrs, X)

    # --- Initialize variables
    for (idx, coef) in enumerate(lasso_soln)
        if column_generation
            if coef != 0
                new_var = add_beta!(model, idx, sign(coef))
                setvalue(new_var, coef)
            end
        else
            add_beta!(model, idx, 1)
            add_beta!(model, idx, -1)
        end
    end
    return model
end


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


function dantzig_lp(y, X, delta;
                    constraint_generation = true,
                    column_generation = true,
                    return_diagnostics = false,
                    verbose = false,
                    timeout = Inf)
    model = initialize_model(
        X, y, minimum(delta), constraint_generation, column_generation, verbose)

    if isa(delta, Array)
        solutions = []
        for d in delta
            for constr in model.linf_pos_constrs
                JuMP.setRHS(constr, d)
            end
            for constr in model.linf_neg_constrs
                JuMP.setRHS(constr, -d)
            end
            solution = solve_model(model, d,
                                   constraint_generation, column_generation,
                                   verbose, timeout)
            push!(solutions, solution)
        end
        return solutions
    else
        return solve_model(model, delta,
                           constraint_generation, column_generation,
                           verbose, timeout)
    end
end


function solve_model(model, delta,
                     constraint_generation, column_generation, verbose, timeout)
    # Logging vars
    start_time = time_ns()
    columns_generated = 0
    constraints_generated = 0
    gurobi_seconds = 0

    solve_status, solve_time = @timed solve(model.gurobi_model)
    gurobi_seconds += solve_time
    status = :InProgress
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        if (time_ns() - start_time) / 1.0e9 > timeout
            status = :Timeout
            break
        end

        # --- Constraint generation ---
        if constraint_generation
            new_constrs = generate_constraints!(model, delta)
            constraints_generated += length(new_constrs)
            while length(new_constrs) > 0
                solve_status, solve_time = @timed solve(model.gurobi_model)
                gurobi_seconds += solve_time
                new_constrs = generate_constraints!(model, delta)
                constraints_generated += length(new_constrs)
            end
        end

        # --- Column generation ---
        if column_generation
            new_var_index, new_var_sign = generate_column(model)
            if new_var_index == nothing
                status = :Optimal
            else
                if verbose
                    info("Column generation iteration $columns_generated:"
                         * "adding beta $new_var_index")
                end
                new_var = add_beta!(model, new_var_index, new_var_sign)
                solve_status, solve_time = @timed solve(model.gurobi_model)
                gurobi_seconds += solve_time
                columns_generated += 1
            end
        else
            status = :Optimal
        end
    end

    beta_values = spzeros(size(model.X)[2])
    for (var, i) in zip(model.pos_betas, model.pos_beta_indices)
        beta_values[i] = getvalue(var)
    end

    for (var, i) in zip(model.neg_betas, model.neg_beta_indices)
        beta_values[i] -= getvalue(var)
    end

    end_time = time_ns()
    # total_seconds = (end_time - start_time) / 1.0e9
    # correct_lasso_vars =
    #     length(intersect(beta_values.nzind, lasso_soln.nzind))
    # total_vars = length(beta_values.nzind)
    # diagnostics = dantzig_diagnostics(total_seconds,
    #                                   lasso_seconds,
    #                                   gurobi_seconds,
    #                                   columns_generated,
    #                                   constraints_generated,
    #                                   correct_lasso_vars,
    #                                   total_vars,
    #                                   string(status))
    diagnostics = nothing  # TODO Fix the diagnostics
    return (model, beta_values, diagnostics)
end


function generate_column(model)
    TOL = 1e-12
    pos_costs, neg_costs = get_reduced_costs(model)
    pos_costs[model.pos_beta_indices] = Inf
    neg_costs[model.neg_beta_indices] = Inf

    pos_min_index = indmin(pos_costs)
    neg_min_index = indmin(neg_costs)
    pos_min_cost = pos_costs[pos_min_index]
    neg_min_cost = neg_costs[neg_min_index]

    min_cost = min(pos_min_cost, neg_min_cost)
    min_sign = ifelse(pos_min_cost < neg_min_cost, 1, -1)
    min_index = ifelse(pos_min_cost < neg_min_cost,
                       pos_min_index, neg_min_index)

    if min_cost < -TOL
        return (min_index, min_sign)
    else
        return (nothing, nothing)
    end
end


function get_reduced_costs(model)
    n, p = size(model.X)
    I_p = speye(p)

    # Compute reduced costs separately for the positive and negative
    # components of Beta
    pos_reduced_costs =
        1 - transpose(model.gurobi_model.linconstrDuals[1:n]) * -model.X
    neg_reduced_costs =
        1 - transpose(model.gurobi_model.linconstrDuals[1:n]) * model.X
    return (pos_reduced_costs, neg_reduced_costs)
end


function generate_constraints!(model, delta;
                               max_constraints = 50, verbose = false)
    Xt = model.X'
    TOL = 1e-6
    constraint_values = Xt * getvalue(model.residuals)
    constraint_indices = sortperm(abs.(constraint_values), rev = true)

    new_constrs = []
    for row in constraint_indices[1:max_constraints]
        val = constraint_values[row]
        if val < -delta - TOL
            # if verbose info("Constraint violated! val = $val") end
            new_constr = @constraint(model.gurobi_model,
                                     dot(Xt[row, :], model.residuals) >= -delta)
            push!(model.linf_neg_constrs, new_constr)
            push!(new_constrs, new_constr)
        elseif val > delta + TOL
            # if verbose info("Constraint violated! val = $val") end
            new_constr = @constraint(model.gurobi_model,
                                     dot(Xt[row, :], model.residuals) <= delta)
            push!(model.linf_pos_constrs, new_constr)
            push!(new_constrs, new_constr)
        end
    end
    return new_constrs
end

# Module end
end





