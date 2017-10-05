module DantzigLP
using JuMP, Gurobi, MathProgBase, Lasso

function dantzig_lp(y, X, delta, method)
    if method == :column_generation
        return colgen_dantzig(y, X, delta)
    elseif method == :simple
        return baseline_dantzig(y, X, delta)
    else
        error("Unrecognized Dantzig solution method.")
    end
end


"""Baseline solution"""
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


function colgen_dantzig(y, X, delta;
                        constraint_generation = true,
                        column_generation = true)
    # TODO difference between custom and standard lasso solns?
    # Initialize model
    # --------------------------------------------------------------------------
    n, p = size(X)
    I_p = speye(p)

    model = Model(solver = GurobiSolver(Method=1))

    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)
    residuals = @variable(model, [1:n])

    abs_beta_constrs = @constraint(model, abs_beta_pos - abs_beta_neg .== 0)
    residual_constrs = @constraint(model, y - residuals .== 0)

    betas = []
    beta_indices = []

    info("Fitting Lasso solution...")
    lasso_soln = vec(fit(LassoPath, X, y, λ = [delta], maxncoef = p).coefs)
    # lasso_soln = sparse(vec(Lasso_soln_delta(y, X, delta)))

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

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    # Generate BFS from lasso solution
    # --------------------------------------------------------------------------
    """
    Closure to add new beta variables to the model.
    WARNING: mutates betas and beta_indices.
    """
    function add_beta(idx)
        if idx in beta_indices
            return nothing
        else
            new_var = @variable(
                model,
                objective = 0,
                inconstraints = vcat(residual_constrs, [abs_beta_constrs[idx]]),
                coefficients = full(vcat(-X[:, idx], [-1.0])))
            push!(beta_indices, idx)
            push!(betas, new_var)
            return new_var
        end
    end

    for (idx, beta) in enumerate(lasso_soln)
        if beta != 0 || column_generation == false
            new_var = add_beta(idx)
            setvalue(new_var, beta)
        end
    end


    # Column / constraint generation
    # --------------------------------------------------------------------------
    solve(model)
    status = :InProgress
    i = 1
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        # --- Constraint generation ---
        if constraint_generation
            new_constrs = generate_constraints(model, delta, X, residuals)
            while length(new_constrs) > 0
                solve(model)
                new_constrs = generate_constraints(model, delta, X, residuals)
            end
        end

        # --- Column generation ---
        if column_generation
            new_var_index = generate_column(model, X, beta_indices)
            if new_var_index == nothing
                status = :Optimal
            else
                info("Column generation iteration $i:"
                     * "adding beta $new_var_index")
                new_var = add_beta(new_var_index)
                solve(model)
                i = i + 1
            end
        else
            status = :Optimal
        end
    end

    beta_values = spzeros(p)
    for (var, i) in zip(betas, beta_indices)
        beta_values[i] = getvalue(var)
    end

    return (model, beta_values, lasso_soln)
end


function generate_column(model, X, beta_indices)
    TOL = 1e-12
    red_costs = get_reduced_costs(model, X)
    red_costs[beta_indices] = Inf

    min_index = indmin(red_costs)
    info(red_costs[min_index])

    if red_costs[min_index] < -TOL
        return min_index
    else
        return nothing
    end
end


function get_reduced_costs(model, X)
    n, p = size(X)
    I_p = speye(p)
    - transpose(model.linconstrDuals[1:p + n]) * vcat(-I_p, -X)
    return - transpose(model.linconstrDuals[1:p + n]) * vcat(-I_p, -X)
end


function generate_constraints(model, delta, X, residuals, max_constraints = 50)
    Xt = X'
    TOL = 1e-6
    constraint_values = Xt * getvalue(residuals)
    constraint_indices = sortperm(abs(constraint_values), rev = true)

    new_constrs = []

    for row in constraint_indices[1:max_constraints]
        val = constraint_values[row]
        if val < -delta - TOL
            info("Constraint violated! val = $val")
            new_constr =
                @constraint(model, dot(Xt[row, :], residuals) >= -delta)
            push!(new_constrs, new_constr)
        elseif val > delta + TOL
            info("Constraint violated! val = $val")
            new_constr = @constraint(model, dot(Xt[row, :], residuals) <= delta)
            push!(new_constrs, new_constr)
        end
    end
    return new_constrs
end


function Lasso_soln_delta(Y, X, delta, TOL=0.0001, maxiter=10000)
    n,p = size(X)
    aa= svds(X,nsv=1,ritzvec=false)[1]
    Lip = aa.S[1]
    Lip = Lip^2

    betak = zeros(p,1)
    grad = -X'*(Y - X*betak)
    err=0

    for iter in 1:maxiter
        betakold = betak
        betak = betak - grad/Lip
        betak = sign.(betak) .* max(abs(betak) - delta/Lip, 0)
        err = norm(betak - betakold,Inf)
        grad = -X'*(Y - X*betak)
        if (err < TOL)
            break
        end
    end
    warn("Lasso solution did not converge within maxiter.")
    return betak
end

# Module end
end





