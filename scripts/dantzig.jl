import DantzigLP
@everywhere using DantzigLP, DataFrames, ProgressMeter, RCall, IterTools, CSV

@everywhere function psm(X, y, λ; nlambda=1000)
    result = R"fastclime::dantzig($X, $y, $λ, $nlambda)"
    return rcopy(result)
end

@everywhere function range_bins(a, b, bins)
    bin_size = (b - a) / bins
    return a .+ collect(0:bins) .* bin_size
end

ns = [200, 500, 1000]
ps = [500, 1000, 5000, 10000]
rhos_ss = [(0, 0), (0.4, 0), (0.8, 0), (0, 0.4), (0, 0.8), (0, 0.95)]
is = collect(1:20)

srand(798)
results = []

params = product(ns, ps, rhos_ss, is) |> collect |> shuffle;
results = @parallel vcat for param in params
    @show param
    instance_id = hash(param)
    (n, p, (rho, s), i) = param

    X, y, actual = DantzigLP.regression_example(
        n, p, 0.1; rho=rho, sparsity=s, standardize=true)
    actual_y = X * actual

    max_lambda = maximum(abs.(X'y))
    min_lambda = norm((X') * (y - X * actual), Inf)
    all_lambda = range_bins(min_lambda, max_lambda, 50)
    single_seconds = @elapsed DantzigLP.dantzig_lp(X, y, min_lambda)
    psm_seconds = @elapsed psm(X, y, min_lambda)
    path_seconds = @elapsed dantzig_lp(X, y, all_lambda)

    df = DataFrame(algo = ["Dantzig LP (Single Lambda)",
                           "Dantzig LP (Lambda Path)",
                           "PSM"],
                   seconds = [single_seconds,
                              path_seconds,
                              psm_seconds],
                   n = n, p = p, rho = rho, s = s, i = instance_id,
                   lambda = min_lambda)
    df
end

CSV.write("dantzig_lp_results.csv", results)
