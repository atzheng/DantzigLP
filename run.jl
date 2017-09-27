using StatsBase, Gadfly, DataFrames


"""
Generate an example dataset of `n` points with `breaks` discontinuities.
Both true values and noise are generated from random normal distributions.
"""
function generate_1D(n, breaks)
    # Generate random indices for the discontinuities
    break_indices = vcat(1, sort(sample(2:n, breaks)), n + 1)
    break_lengths = break_indices[2:end] - break_indices[1:end - 1]

    # Generate random levels for each discontinuity
    break_levels = randn(breaks + 1)
    segments = [repeat([x[2]], inner = x[1])
                for x in zip(break_lengths, break_levels)]

    clean_signal = vcat(segments...)
    noisy_signal = clean_signal + randn(n)
    graph_row = vcat([-1, 1], sparsevec(zeros(n - 2)))
    graph = transpose(hcat([circshift(graph_row, i - 1)
                            for i in 1:n - 1]...))
    return clean_signal, noisy_signal, graph
end

function generate_regression(n, breaks)
    # Generate random indices for the discontinuities
    break_indices = vcat(1, sort(sample(2:n, breaks)), n + 1)
    break_lengths = break_indices[2:end] - break_indices[1:end - 1]

    # Generate random levels for each discontinuity
    break_levels = randn(breaks + 1)
    segments = [repeat([x[2]], inner = x[1])
                for x in zip(break_lengths, break_levels)]

    clean_signal = vcat(segments...)
    noisy_signal = clean_signal + randn(n)
    graph_row = vcat([-1, 1], sparsevec(zeros(n - 2)))
    graph = transpose(hcat([circshift(graph_row, i - 1)
                            for i in 1:n - 1]...))
    return clean_signal, noisy_signal, graph
end

function plot_example(clean, noisy, fit)
    to_plot = DataFrame(x = 1:length(clean),
                        clean = clean,
                        noisy = noisy,
                        fit = fit)
    plot(stack(to_plot, [:clean, :noisy, :fit]),
         x = :x, y = :value, color = :variable, alpha = 0.3,
         Geom.line)
end

clean, noisy, graph = generate_1D(1000, 2)
lambda = 6
m, denoised = slow_flsa(noisy, lambda)
plot_example(clean, noisy, denoised)
cbasis, rbasis = MathProgBase.getbasis(internalmodel(m))


random_graph = randn(1000, 1000) |> round


function time_instance(algos, signal, lambda)
    return [@timed(algo(signal, lambda)) for algo in algos]
end

x = time_instance([slow_flsa, colgen_flsa], noisy, lambda)


x = @time(slow_flsa(noisy, lambda, random_graph))


model, slow = slow_flsa(noisy, lambda, graph)
model2, col = colgen_flsa(noisy, lambda, graph)

m = Model()
@constraint(m, con, 0 <= 1)
@objective(m, Max, 0)
@variable(m, 0 <= x <= 1, objective = 5, inconstraints = [con], coefficients = [1.0])
@variable(m, 0 <= y <= 1, objective = 1, inconstraints = [con], coefficients = [1.0])
@variable(m, objective = 10, inconstraints = [con], coefficients = [1.0])
@variable(m, objective = 10, inconstraints = [m.linconstr[1]], coefficients = [1.0])

solve(m)







