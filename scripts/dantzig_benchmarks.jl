using DataFrames

ns = 2 .^ (7:12)
ps = 2 .^ (7:12)
rhos = 0.0:0.2:0.6
delta_pcts = 0.2:0.2:0.8


function run_benchmark(n, p, rho)
    X, y, _ = generate_regression_example(n, p, 0.2, rho = rho)
    return vcat([run_instance(X, y, pct, colgen, congen)
                 for pct in delta_pcts
                 for colgen in [false, true]
                 for congen in [false, true]]...)
end


function run_instance(X, y, pct, colgen, congen)
    n, p = size(X)
    max_delta = maximum(abs.(X'y))
     _, seconds = @timed DantzigLP.dantzig_lp(y, X, max_delta * pct,
                                              column_generation = colgen,
                                              constraint_generation = congen)
    return DataFrame(n = n, p = p, delta_pct = pct,
                     colgen = colgen, congen = congen, seconds = seconds)
end

results = vcat([run_benchmark(n, p, rho)
                for n in ns for p in ps for rho in rhos]...)


run_instance(X, y, 0.8, true, true)

