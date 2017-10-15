using DataFrames, ProgressMeter

include("../utils.jl")
include("../dantzig.jl")

ns = 2 .^ (7:11)
ps = 2 .^ (7:11)
rhos = 0.0:0.2:0.6
delta_pcts = 0.2:0.2:0.8


function run_benchmark(n, p, rho)
    info("*** Running benchmark with n=$n, p=$p, rho=$rho ***")
    X, y, _ = generate_regression_example(n, p, 0.2, rho = rho)
    return vcat([run_instance(X, y, pct, rho, colgen, congen)
                 for pct in delta_pcts
                 for colgen in [false, true]
                 for congen in [false, true]]...)
end


function run_instance(X, y, pct, rho, colgen, congen)
    n, p = size(X)
    max_delta = maximum(abs.(X'y))
    _, _, diagnostics = DantzigLP.dantzig_lp(y, X, max_delta * pct,
                                             column_generation = colgen,
                                             constraint_generation = congen,
                                             return_diagnostics = true)
    params_df = DataFrame(n = n, p = p, delta_pct = pct, rho = rho,
                          colgen = colgen, congen = congen)
    results_df = struct2df([diagnostics])
    return hcat(params_df, results_df)
end

# Benchmark vs. other column / constraint generation settings
params = [(n, p, rho) for n in ns for p in ps for rho in rhos]
results = []
@showprogress for param in params
    push!(results, run_benchmark(param...))
end
results_df = vcat(results...)
writetable("results.csv", results_df)


n = 5000
p = 50000
X, y, _ = generate_regression_example(n, p, 0.2)
results = run_instance(X, y, 0.8, 0.0, true, true)





