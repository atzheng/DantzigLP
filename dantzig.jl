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
function baseline_dantzig(y, X, delta = 6)
    n, p = size(X)
    Xty = transpose(X) * y
    XtX = transpose(X) * X

    model = Model(solver = GurobiSolver())

    Beta = @variable(model, [1:p])
    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)

    linf_pos_constrs = @constraint(model, Xty - XtX * Beta .<= delta)
    linf_neg_constrs = @constraint(model, Xty - XtX * Beta .>= -delta)
    abs_beta_constrs = @constraint(model, abs_beta_pos - abs_beta_neg .== Beta)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    solve(model)
    return model, sparse(getvalue(Beta))
end


function colgen_dantzig(y, X, delta = 6;
                        constraint_generation = true,
                        column_generation = true)
    # TODO difference between custom and standard lasso solns?
    # Initialize model
    # --------------------------------------------------------------------------
    n, p = size(X)
    Xty = transpose(X) * y
    XtX = transpose(X) * X
    I_p = speye(p)

    model = Model(solver = GurobiSolver())

    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)
    abs_beta_constrs = @constraint(model, abs_beta_pos - abs_beta_neg .== 0)

    betas = []
    beta_indices = []

    # --- Initialize constraints ---
    if constraint_generation == false
        init_constrs = 1:p
    else
        # initialize with random constraints
        init_constrs = unique(sample(1:p, 10))
    end

    linf_pos_constrs = @constraint(model, Xty[init_constrs] .<= delta)
    linf_neg_constrs = @constraint(model, Xty[init_constrs] .>= -delta)
    constraints = vcat(linf_pos_constrs, linf_neg_constrs)
    constraint_indices = vcat(init_constrs, init_constrs)

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
                inconstraints = vcat(constraints,
                                     [abs_beta_constrs[idx]]),
                coefficients = full(vcat(-XtX[constraint_indices, idx],
                                         [-1.0])))
            push!(beta_indices, idx)
            push!(betas, new_var)
            return new_var
        end
    end

    info("Fitting Lasso solution...")
    # lasso_soln = fit(LassoPath, X, y, λ = [delta], maxncoef = p).coefs
    lasso_soln = sparse(Lasso_soln_delta(y, X, delta))
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
            new_constrs, new_constr_indices = generate_constraints(
                model, Xty, XtX, betas, beta_indices, delta, constraint_indices)
            info("Generating new constraints: $new_constr_indices")
            append!(constraints, new_constrs)
            append!(constraint_indices, new_constr_indices)

            while length(new_constrs) > 0
                solve(model)
                new_constrs, new_constr_indices = generate_constraints(
                    model, Xty, XtX, betas, beta_indices, delta,
                    constraint_indices)
                append!(constraints, new_constrs)
                append!(constraint_indices, new_constr_indices)
                info("Generating new constraints: $new_constr_indices")
            end
        end

        # --- Column generation ---
        if column_generation
            new_var_index =
                generate_column(model, XtX, beta_indices, constraint_indices)
            if new_var_index == nothing
                status = :Optimal
            else
                info("Column generation iteration $i:"
                     * "adding beta $new_var_index")
                new_var = add_beta(new_var_index)
                solve(model)
                i = i + 1
            end
        end
    end

    beta_values = spzeros(p)
    for (var, i) in zip(betas, beta_indices)
        beta_values[i] = getvalue(var)
    end

    return (model, beta_values)
end


function generate_column(model, XtX, beta_indices, constraint_indices)
    TOL = 1e-12  # Numerical tolerance TODO How to set this?
    I_p = speye(size(XtX)[1])

    red_costs = - transpose(model.linconstrDuals) *
        vcat(-XtX[constraint_indices, :], -I_p)

    red_costs[beta_indices] = Inf
    min_index = indmin(red_costs)

    if red_costs[min_index] < -TOL
        return min_index
    else
        return nothing
    end
end


function generate_constraints(model, Xty, XtX, beta_vars, beta_indices, delta,
                              existing_constraints)
    TOL = 1e-11
    constraint_values = Xty - XtX[:, beta_indices] * getvalue(beta_vars)
    new_constrs = []
    new_constr_indices = []

    for (row, val) in enumerate(constraint_values)
        if val < -delta - TOL
            info("Constraint violated! val = $val")
            new_constr = @constraint(
                model,
                Xty[row] - dot(XtX[row, beta_indices], beta_vars) >= -delta)
            push!(new_constrs, new_constr)
            push!(new_constr_indices, row)
        elseif val > delta + TOL
            info("Constraint violated! val = $val")
            new_constr = @constraint(
                model,
                Xty[row] - dot(XtX[row, beta_indices], beta_vars) <= delta)
            push!(new_constrs, new_constr)
            push!(new_constr_indices, row)
        end
    end
    return new_constrs, new_constr_indices
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





