@everywhere using DantzigLP, IterTools, ProgressMeter, CSV

@everywhere function run_instance(n, p)
    X, y, β, α = DantzigLP.fused_regression_example(n, p, 20)
    λ = norm((X') * (y - X * β), Inf)

    bl_time = @elapsed model, βdantzig =
        DantzigLP.baseline_dantzig_fused_regression(X, y, λ; timeout=300)
    cg_time = @elapsed βbp, model, diagnostics =
        DantzigLP.dantzig_fused_regression(
            X, y, λ; max_columns=40, verbose=false,
            column_generation=false)

    diagnostics[:baseline_secs] = bl_time
    diagnostics[:total_secs] = cg_time
    return diagnostics
end

ns = [200, 500, 1000]
ps = [500, 1000, 5000, 10000]
instances = collect(1:20)

params = product(ns, ps, instances) |> collect |> shuffle

srand(798)
results = @parallel vcat for param in params
    gc()
    @show param
    instance_id = hash(param)
    n, p, i = param

    diagnostics = run_instance(n, p)
    diagnostics[:instance_id] = instance_id
    diagnostics[:n] = n
    diagnostics[:p] = p

    diagnostics
end

CSV.write("fused_regression.csv", results)
