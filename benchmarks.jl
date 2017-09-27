using JuMP, Gurobi


"""
Naive LP implementation of FLSA.
"""
function slow_flsa(signal, lambda, graph)
    # Instantiate problem
    m = Model(solver = GurobiSolver())
    n = length(signal)
    graph_n = size(graph)[1]  # First dimension of the graph D

    B = @variable(m, [1:n])

    # Absolute error compared to data vector y
    error_pos = @variable(m, [1:n], lowerbound = 0)
    error_neg = @variable(m, [1:n], lowerbound = 0)

    # Graph errors
    diff_pos = @variable(m, [1:graph_n], lowerbound = 0)
    diff_neg = @variable(m, [1:graph_n], lowerbound = 0)

    # Constraints enforcing absolute values
    error_constr = @constraint(
        m, [i = 1:n], error_pos[i] - error_neg[i] == signal[i] - B[i])
    diff_constr = @constraint(
        m, [i = 1:n - 1],
        diff_pos[i] - diff_neg[i]
        == sum([graph[i, j] * B[j] for j in 1:length(signal)]))

    obj = @objective(m, Min,
                     sum(error_pos[1:n] + error_neg[1:n])
                     + lambda * sum(diff_pos[1:n - 1] + diff_neg[1:n - 1]))

    # Solve and return solution
    status = solve(m)
    return (m, getvalue(B))
end


"""
Naive LP implementation of the lasso problem.
"""
function slow_lasso(signal, lambda)
    # Instantiate problem
    m = Model(solver = GurobiSolver())
    n = length(signal)

    @variable(m, B_pos[1:n] >= 0)
    @variable(m, B_neg[1:n] >= 0)

    @variable(m, error_pos[1:n] >= 0)
    @variable(m, error_neg[1:n] >= 0)

    @constraint(m, error_constr[i=1:n],
                error_pos[i] - error_neg[i] == signal[i] - B_pos[i] + B_neg[i])

    @objective(m, Min,
               sum(error_pos[1:n] + error_neg[1:n])
               + lambda * sum(B_pos[1:n] + B_neg[1:n]))

    # Solve and return solution
    status = solve(m)
    return getValue(B_pos)
end




