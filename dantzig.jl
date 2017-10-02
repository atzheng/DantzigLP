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


function colgen_dantzig(y, X, delta = 6)
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

    linf_pos_constrs = @constraint(model, Xty .<= delta)
    linf_neg_constrs = @constraint(model, Xty .>= -delta)
    abs_beta_constrs = @constraint(model, abs_beta_pos - abs_beta_neg .== 0)
    all_constrs = vcat(linf_pos_constrs, linf_neg_constrs, abs_beta_constrs)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    # Generate BFS from lasso solution
    # --------------------------------------------------------------------------
    betas = []
    beta_indices = []

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
                inconstraints = vcat(linf_pos_constrs,
                                     linf_neg_constrs,
                                     [abs_beta_constrs[idx]]),
                coefficients = full(vcat(-XtX[:, idx],
                                         -XtX[:, idx],
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
        if beta != 0
            new_var = add_beta(idx)
            setvalue(new_var, beta)
        end
    end


    # Column generation
    # --------------------------------------------------------------------------
    solve(model)
    status = :InProgress
    i = 1
    while status == :InProgress
        new_var_index = generate_column(model, XtX, beta_indices)
        if new_var_index == nothing
            status = :Optimal
        else
            @printf("**Column generation iteration %d: adding beta %d**\n",
                    i, new_var_index)
            new_var = add_beta(new_var_index)
            solve(model)
            i = i + 1
        end
    end

    beta_values = spzeros(p)
    for (var, i) in zip(betas, beta_indices)
        beta_values[i] = getvalue(var)
    end

    return (model, beta_values)
end


function generate_column(model, XtX, beta_indices)
    TOL = 1e-16  # Numerical tolerance TODO How to set this?
    I_p = speye(size(XtX)[1])

    imodel = internalmodel(model)
    p = MathProgBase.getconstrduals(imodel)
    red_costs = - transpose(p) * vcat(XtX, -XtX, -I_p)
    red_costs[beta_indices] = Inf
    min_index = indmin(red_costs)

    if red_costs[min_index] < -TOL
        return min_index
    else
        return nothing
    end
end


function Lasso_soln_delta(Y, X, delta, TOL=0.0001, maxiter=10000)
    n,p = size(X);
    aa= svds(X,nsv=1,ritzvec=false)[1]
    Lip = aa.S[1]
    Lip = Lip^2

    betak = zeros(p,1);
    grad = -X'*(Y - X*betak);
    err=0;

    for iter in 1:maxiter
        betakold = betak;
        betak = betak - grad/Lip;
        betak = sign.(betak) .* max(abs(betak) - delta/Lip, 0);
        err = norm(betak - betakold,Inf);
        grad = -X'*(Y - X*betak);
        if (err < TOL)
            break;
        end
    end
    warn("Lasso solution did not converge within maxiter.")
    return betak
end

# Module end
end





