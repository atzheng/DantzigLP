using JuMP, Gurobi, MathProgBase


function colgen_regression(y, X, D, lambda)
    m = Model(solver = GurobiSolver())
    n = length(signal)

    error_pos = @variable(m, [1:n], lowerbound = 0)
    error_neg = @variable(m, [1:n], lowerbound = 0)

    delta_pos = @variable(m, [1:n], lowerbound = 0)
    delta_neg = @variable(m, [1:n], lowerbound = 0)

    error_constr = @constraint(m, [i = 1:n],
                               error_pos[i] - error_neg[i] == y[i])
    delta_constr = @constraint(m, [i = 1:n],
                               delta_pos[i] - delta_neg[i] == 0)

    obj = @objective(m, Min,
                     sum(error_pos + error_neg)
                     + lambda * sum(delta_pos + delta_neg))

    solve(m)
    status = :InProgress
    i = 1
    while status == :InProgress
        @printf("**Column generation iteration %d**\n", i)
        status, m = get_new_basis(lambda, m, error_constr, diff_constr)
        solve(m)
        i = i + 1
    end

    return (m, getvalue(B))
end


function colgen_flsa(signal, lambda, graph)
    # Instantiate problem
    m = Model(solver = GurobiSolver())
    n = length(signal)

    B = @variable(m, [1:n])

    # Absolute error compared to data vector y
    error_pos = @variable(m, [1:n], lowerbound = 0)
    error_neg = @variable(m, [1:n], lowerbound = 0)

    # Absolute difference from B_k+1 to B_k
    # diff_pos = @variable(m, [1:n - 1], lowerbound = 0)
    # diff_neg = @variable(m, [1:n - 1], lowerbound = 0)

    error_constr = @constraint(m, [i = 1:n], error_pos[i] - error_neg[i]
                               == - B[i] + signal[i])
    diff_constr = @constraint(
        m, [i = 1:size(graph)[1]],
        0 == sum([graph[i, j] * B[j] for j in 1:length(signal)]))

    obj = @objective(m, Min, sum(error_pos + error_neg))

    # # Solve and return solution
    solve(m)
    status = :InProgress
    i = 1
    while status == :InProgress
        @printf("**Column generation iteration %d**\n", i)
        status, m = get_new_basis(lambda, m, error_constr, diff_constr)
        solve(m)
        i = i + 1
    end
    return (m, getvalue(B))
end


function get_new_basis(lambda, model, error_constr, diff_constr)
    m = internalmodel(model)
    p = MathProgBase.getconstrduals(m)
    n = div(length(p), 2) + 1
    cbasis, rbasis = MathProgBase.getbasis(m)

    new_var_type, new_var_idx = find_negative_reduced_cost(lambda, p)

    if new_var_type == :None return (:Optimal, model)
        # Probably never reach this; Initial BFS is beta
        # elseif new_var_type == :Beta
        # @defVar(model, objective=0,
        #         inconstraints=model.linconstr[[new_var_idx,
        #                                        new_var_idx]],
        #         coefficients=Anew
        #         )
    elseif new_var_type == :Error_Pos
        new_var = @variable(model,
                            objective = 1,
                            inconstraints = [error_constr[new_var_idx]],
                            coefficients = [1.0])
    elseif new_var_type == :Error_Neg
        new_var = @variable(model,
                            objective = 1,
                            inconstraints = [error_constr[new_var_idx]],
                            coefficients = [-1.0])
    elseif new_var_type == :Diff_Pos
        new_var = @variable(model,
                            objective = lambda,
                            inconstraints = [diff_constr[new_var_idx]],
                            coefficients = [1.0])
    elseif new_var_type == :Diff_Neg
        new_var = @variable(model,
                            objective = lambda,
                            inconstraints = [diff_constr[new_var_idx]],
                            coefficients = [-1.0])
    end
    return (:InProgress, model)
end


"""
This function works exclusively for the D^1 case and depends on variable order.
TODO: think about randomizing search order.
"""
function find_negative_reduced_cost(lambda, p)
    m = length(p)
    n = div(m, 2) + 1

    for i in 1:n
        # Betas:
        pAj = (p[i]
               + (if (i < n) p[n + i] else 0 end)
               - (if (i > 1) p[n + i - 1] else 0 end))
        redcost = - pAj
        if redcost < 0 return (:Beta, i) end

        if 1 + p[i] < 0 return (:Error_Pos, i)  # l^+
        elseif 1 - p[i] < 0 return (:Error_Neg, i)  # l^-
        elseif (i < n) && (lambda + p[i] < 0) return (:Diff_Pos, i)  # delta^+
        elseif (i < n) && (lambda - p[i] < 0) return (:Diff_Neg, i)  # delta^-
        end
    end
    # No negative reduced costs; return negative index indicating optimality
    return (:None, -1)
end




    # n = length(p) + 1
    # pAj = vcat([p[i]
    #             # + ((i >= div(n, 2)) ? 0 : p[div(n, 2) + i])
    #             # - (if (i > 1) p[div(n, 2) + i - 1] else 0 end)
    #             for i in 1:div(n, 2)],  # \beta
    #            p[1:div(n, 2)],  # l^+
    #            -p[1:div(n, 2)],  # l^-
    #            p[div(n, 2) + 1:n - 1],  # \delta^+
    #            p[div(n, 2) + 1:n - 1])  # \delta^-
    # reduced_costs = m
