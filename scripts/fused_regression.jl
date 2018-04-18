using DantzigLP, CSV
include("utils.jl")

config = parse_config(ARGS[1])
n = config[:n]
p = config[:p]

# Warmup
X, y, β, α = DantzigLP.fused_regression_example(200, 200, 20)
λ = 0.1
DantzigLP.baseline_dantzig_fused_regression(X, y, λ; timeout=300)
DantzigLP.dantzig_fused_regression(
    X, y, λ; max_columns=40, verbose=false,
    column_generation=false)


# Benchmarks
X, y, β, α = DantzigLP.fused_regression_example(n, p, 20)
λ = norm((X') * (y - X * β), Inf)
bl_time = @elapsed model, βdantzig =
    DantzigLP.baseline_dantzig_fused_regression(X, y, λ; Method=1)
cg_time = @elapsed βbp, model, diagnostics =
    DantzigLP.dantzig_fused_regression(
        X, y, λ; max_columns=40, verbose=false,
        column_generation=false, solver_params=Dict(:Method=>1))

diagnostics[:baseline_secs] = bl_time
diagnostics[:total_secs] = cg_time

instance_id = hash(config)
diagnostics[:instance_id] = instance_id
diagnostics[:n] = n
diagnostics[:p] = p

CSV.write(@sprintf("fused_regression/results/%s.csv", ARGS[1]), diagnostics)
