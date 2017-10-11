using JuMP, Gurobi, MathProgBase


"""Baseline solution"""
function baseline_regression(y, X, D = nothing, lambda=6)
    n, m = size(X)
    if D == nothing
        D = speye(m)
    end

    k = size(D)[1]

    model = Model(solver = GurobiSolver())

    Beta = @variable(model, [1:m])

    error_pos = @variable(model, [1:n], lowerbound = 0)
    error_neg = @variable(model, [1:n], lowerbound = 0)

    delta_pos = @variable(model, [1:k], lowerbound = 0)
    delta_neg = @variable(model, [1:k], lowerbound = 0)

    error_constr = @constraint(model, [i = 1:n],
                               error_pos[i] - error_neg[i]
                               == y[i] - sum(X[i, :] .* Beta))
    delta_constr = @constraint(model, [i = 1:k],
                               delta_pos[i] - delta_neg[i]
                               == sum(D[i, :] .* Beta))

    obj = @objective(model, Min,
                     sum(error_pos + error_neg)
                     + lambda * sum(delta_pos + delta_neg))

    solve(model)
    return model, getvalue(Beta)
end


"""Column generation solution"""
function colgen_regression(y, X, D = nothing, lambda=6)
    model = Model(solver = GurobiSolver())
    n, m = size(X)
    if D == nothing
        D = speye(m)
    end

    k = size(D)[1]

    error_pos = @variable(model, [1:n], lowerbound = 0)
    error_neg = @variable(model, [1:n], lowerbound = 0)

    delta_pos = @variable(model, [1:k], lowerbound = 0)
    delta_neg = @variable(model, [1:k], lowerbound = 0)

    error_constr = @constraint(model, [i = 1:n],
                               error_pos[i] - error_neg[i] == y[i])
    delta_constr = @constraint(model, [i = 1:k],
                               delta_pos[i] - delta_neg[i] == 0)

    obj = @objective(model, Min,
                     sum(error_pos + error_neg)
                     + lambda * sum(delta_pos + delta_neg))

    # We generate the Beta and delta variables dynamically
    betas = []
    beta_indices = []
    delta_indices = []

    solve(model)
    status = :InProgress
    i = 1
    while status == :InProgress
        new_var, new_var_index, new_delta_indices = get_regression_basis(
            lambda, X, D, model, beta_indices, delta_indices,
            error_constr, delta_constr)

        if new_var_index in beta_indices
            error("Cycling")
        end

        if new_var == nothing
            status = :Optimal
        else
            @printf("**Column generation iteration %d: adding beta %d**\n",
                    i, new_var_index)
            push!(betas, new_var)
            push!(beta_indices, new_var_index)
            delta_indices = vcat(delta_indices, new_delta_indices)
            solve(model)
            i = i + 1
        end
    end

    beta_values = zeros(m)
    for x in zip(betas, beta_indices)
        var, i = x
        beta_values[i] = getvalue(var)
    end

    return (model, beta_values)
end


function get_regression_basis(lambda, X, D, model,
                              existing_betas, existing_deltas,
                              error_constr, delta_constr)
    imodel = internalmodel(model)
    p = MathProgBase.getconstrduals(imodel)
    new_var_type, new_var_idx = find_negative_reduced_cost(
        lambda, X, D, existing_betas, p)

    if new_var_type == :None
        return (nothing, nothing, nothing)
    elseif new_var_type == :Beta
        new_var = @variable(
            model,
            objective = 0,
            inconstraints = vcat(error_constr, delta_constr),
            coefficients = full(vcat(X[:, new_var_idx], -D[:, new_var_idx])))

        new_deltas = setdiff(find(D[:, new_var_idx] != 0), existing_deltas)

        # for row in new_deltas
        #     new_delta_pos = @variable(
        #         model,
        #         lowerbound = 0,
        #         objective = lambda,
        #         inconstraints = [delta_constr[row]],
        #         coefficients = [1.0]
        #     )

        #     new_delta_neg = @variable(
        #         model,
        #         lowerbound = 0,
        #         objective = lambda,
        #         inconstraints = [delta_constr[row]],
        #         coefficients = [-1.0]
        #     )
        # end

        return (new_var, new_var_idx, new_deltas)
    end
end


function find_negative_reduced_cost(lambda, X, D, existing_betas, p)
    TOL = 1e-11  # Numerical tolerance TODO How to set this?
    rows, cols = size(X)

    red_costs = - transpose(p) * vcat(X, -D)
    red_costs[existing_betas] = Inf
    min_index = indmin(red_costs)

    if red_costs[min_index] < -TOL
        return (:Beta, min_index)
    else
        return (:None, -1)
    end

    # for j in 1:cols
    #     red_cost = - transpose(p) * vcat(X[:, j], -D[:, j])
    #     red_cost::Float64
    #     if red_cost < -1e-11  # TODO How to set numerical tolerance?
    #         print(red_cost)
    #         return (:Beta, j)
    #     end
    # end
    # return (:None, -1)
end
