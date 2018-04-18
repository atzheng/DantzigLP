using DantzigLP, DataFrames, ProgressMeter, RCall, IterTools, CSV

include("utils.jl")
config = parse_config(ARGS[1])

lambda_pct = get(config, :lambdapct, 1)

function psm(X, y, lambda; nlambda=1000)
    result = R"max(system.time(fastclime::dantzig($X, $y, $lambda, $nlambda)))"
    return rcopy(result)
end

function range_bins(a, b, bins)
    bin_size = (b - a) / bins
    return a .+ collect(0:bins) .* bin_size
end

instance_id = hash(config)

# Warmup runs
info("WARMING UP...")
X, y, actual = DantzigLP.regression_example(100, 200, 0.2)
psm(X, y, 1)
DantzigLP.dantzig_lp(X, y, 1)
DantzigLP.baseline_dantzig(X, y, 1, timeout=60)

# Logged runs
X, y, actual = DantzigLP.regression_example(
    config[:n], config[:p], 0.2 * config[:n] / config[:p];
    rho=config[:rho], sparsity=config[:s])
actual_y = X * actual
max_lambda = maximum(abs.(X'y))
min_lambda = norm((X') * (y - X * actual), Inf) * lambda_pct
all_lambda = range_bins(min_lambda, max_lambda, 50)


info("STARTING BENCHMARKS...")
psm_t = psm(X, y, min_lambda)
dlp_single_3_t = @elapsed DantzigLP.dantzig_lp(X, y, min_lambda;
                                               solver_params=Dict(:Method=>-1))
dlp_path_3_t = @elapsed DantzigLP.dantzig_lp(X, y, all_lambda;
                                             solver_params=Dict(:Method=>-1))
dlp_single_1_t = @elapsed DantzigLP.dantzig_lp(X, y, min_lambda;
                                               solver_params=Dict(:Method=>1))
dlp_path_1_t = @elapsed DantzigLP.dantzig_lp(X, y, all_lambda;
                                             solver_params=Dict(:Method=>1))
bl_3_t = @elapsed DantzigLP.baseline_dantzig(X, y, min_lambda;
                                             Method=-1, timeout=300)
bl_1_t = @elapsed m, soln = DantzigLP.baseline_dantzig(X, y, min_lambda;
                                                       Method=1, timeout=300)

results = DataFrame(
    algo = ["PSM", "DantzigLP", "DantzigLP", "DantzigLP", "DantzigLP",
            "Gurobi", "Gurobi"],
    lambda_path = [true, false, true, false, true, false, false],
    solver = ["PSM", "Concurrent", "Concurrent", "Dual Simplex", "Dual Simplex",
              "Concurrent", "Dual Simplex"],
    seconds = [psm_t, dlp_single_3_t, dlp_path_3_t, dlp_single_1_t, dlp_path_1_t,
               bl_3_t, bl_1_t],
    n = config[:n], p = config[:p], rho = config[:rho], s = config[:s],
    i = repr(instance_id),
    lambda_pct = lambda_pct,
    lambda = min_lambda,
    L0 = norm(soln, 0))

CSV.write(@sprintf("dantzig/results/%s.csv", ARGS[1]), results)
