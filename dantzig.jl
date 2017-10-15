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
end

"""
Baseline solution. Mainly for testing.
Under normal conditions, should be identical to dantzig_lp with
column / constraint generation disabled.
"""
function baseline_dantzig(y, X, delta)
    n, p = size(X)

    model = Model(solver = GurobiSolver())

    Beta = @variable(model, [1:p])
    residuals = @variable(model, [1:n])
    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)

    abs_beta_constrs = @constraint(model, abs_beta_pos - abs_beta_neg .== Beta)
    residual_constrs = @constraint(model, y - X * Beta .== residuals)
    linf_pos_constrs = @constraint(model, X' * residuals .<= delta)
    linf_neg_constrs = @constraint(model, X' * residuals .>= -delta)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    solve(model)
    return model, sparse(getvalue(Beta))
end


function dantzig_lp(y, X, delta;
                    constraint_generation = true,
                    column_generation = true,
                    return_diagnostics = false,
                    verbose = false)
    tic()
    # Initialize model
    # --------------------------------------------------------------------------
    n, p = size(X)
    I_p = speye(p)


    output_flag = if verbose 1 else 0 end
    model = Model(solver = GurobiSolver(Method = 1, OutputFlag = output_flag))

    residuals = @variable(model, [1:n])
    residual_constrs = @constraint(model, y - residuals .== 0)

    pos_betas = []
    neg_betas = []
    pos_beta_indices = []
    neg_beta_indices = []

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
        @constraint(model, X'[init_constrs, :] * residuals .<= delta)
    linf_pos_constrs =
        @constraint(model, X'[init_constrs, :] * residuals .>= -delta)

    obj = @objective(model, Min, 0)

    # Generate BFS from lasso solution
    # --------------------------------------------------------------------------
    """
    Closure to add new beta variables to the model.
    WARNING: mutates betas and beta_indices.
    """
    function add_beta(idx, sgn)
        if sgn > 0
            beta_indices = pos_beta_indices
            betas = pos_betas
        else
            beta_indices = neg_beta_indices
            betas = neg_betas
        end

        if idx in beta_indices
            return nothing
        else
            new_var = @variable(
                model,
                objective = 1,
                inconstraints = residual_constrs,
                coefficients = -sgn * X[:, idx],
                lowerbound = 0)
            push!(beta_indices, idx)
            push!(betas, new_var)
            return new_var
        end
    end

    for (idx, beta) in enumerate(lasso_soln)
        if column_generation
            if beta != 0
                new_var = add_beta(idx, sign(beta))
                setvalue(new_var, beta)
            end
        else
            add_beta(idx, 1)
            add_beta(idx, -1)
        end
    end


    # Column / constraint generation
    # --------------------------------------------------------------------------
    columns_generated = 0
    constraints_generated = 0
    gurobi_seconds = 0

    gurobi_seconds += @elapsed solve(model)
    status = :InProgress
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        # --- Constraint generation ---
        if constraint_generation
            new_constrs = generate_constraints(model, delta, X, residuals)
            constraints_generated += length(new_constrs)
            while length(new_constrs) > 0
                gurobi_seconds += @elapsed solve(model)
                new_constrs = generate_constraints(model, delta, X, residuals)
                constraints_generated += length(new_constrs)
            end
        end

        # --- Column generation ---
        if column_generation
            new_var_index, new_var_sign =
                generate_column(model, X, pos_beta_indices, neg_beta_indices)
            if new_var_index == nothing
                status = :Optimal
            else
                if verbose
                    info("Column generation iteration $columns_generated:"
                         * "adding beta $new_var_index")
                end
                new_var = add_beta(new_var_index, new_var_sign)
                gurobi_seconds += @elapsed solve(model)
                columns_generated += 1
            end
        else
            status = :Optimal
        end
    end

    beta_values = spzeros(p)
    for (var, i) in zip(pos_betas, pos_beta_indices)
        beta_values[i] = getvalue(var)
    end

    for (var, i) in zip(neg_betas, neg_beta_indices)
        beta_values[i] -= getvalue(var)
    end

    total_seconds = toc()
    if return_diagnostics
        correct_lasso_vars =
            length(intersect(beta_values.nzind, lasso_soln.nzind))
        total_vars = length(beta_values.nzind)
        diagnostics = dantzig_diagnostics(total_seconds,
                                          lasso_seconds,
                                          gurobi_seconds,
                                          columns_generated,
                                          constraints_generated,
                                          correct_lasso_vars,
                                          total_vars)
        return (model, beta_values, diagnostics)
    else
        return (model, beta_values)
    end
end


function generate_column(model, X, pos_beta_indices, neg_beta_indices)
    TOL = 1e-12
    pos_costs, neg_costs = get_reduced_costs(model, X)
    pos_costs[pos_beta_indices] = Inf
    neg_costs[neg_beta_indices] = Inf

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


function get_reduced_costs(model, X)
    n, p = size(X)
    I_p = speye(p)

    # Compute reduced costs separately for the positive and negative
    # components of Beta
    pos_reduced_costs = 1 - transpose(model.linconstrDuals[1:n]) * -X
    neg_reduced_costs = 1 - transpose(model.linconstrDuals[1:n]) * X
    return (pos_reduced_costs, neg_reduced_costs)
end


function generate_constraints(model, delta, X, residuals;
                              max_constraints = 50, verbose = false)
    Xt = X'
    TOL = 1e-6
    constraint_values = Xt * getvalue(residuals)
    constraint_indices = sortperm(abs.(constraint_values), rev = true)

    new_constrs = []

    for row in constraint_indices[1:max_constraints]
        val = constraint_values[row]
        if val < -delta - TOL
            # if verbose info("Constraint violated! val = $val") end
            new_constr =
                @constraint(model, dot(Xt[row, :], residuals) >= -delta)
            push!(new_constrs, new_constr)
        elseif val > delta + TOL
            # if verbose info("Constraint violated! val = $val") end
            new_constr = @constraint(model, dot(Xt[row, :], residuals) <= delta)
            push!(new_constrs, new_constr)
        end
    end
    return new_constrs
end

# Module end
end





