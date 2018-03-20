@everywhere using DantzigLP, IterTools, ProgressMeter, CSV

@everywhere function run_instance(n, p, compare=false)
    X, y, β = DantzigLP.basis_pursuit_example(n, p, 0.1)

    if compare
        bl_time = @elapsed model, βdantzig = DantzigLP.baseline_basis_pursuit(
            X, y; timeout=180)
    else
        bl_time = nothing
    end

    cg_time = @elapsed βbp, model, diagnostics = DantzigLP.basis_pursuit(
        X, y; max_columns=40, verbose=false)

    diagnostics[:baseline_secs] = bl_time
    diagnostics[:total_secs] = cg_time
    return diagnostics
end

ns = [200, 500, 1000]
ps = [500, 1000, 10000, 100000, 400000]
instances = collect(1:20)

params = product(ns, ps, instances) |> collect |> shuffle

srand(798)
results = @parallel vcat for param in params
    @show param
    instance_id = hash(param)
    n, p, i = param
    # compare = n * p <= 500 * 10000
    compare = true

    diagnostics = run_instance(n, p, compare)
    diagnostics[:instance_id] = instance_id
    diagnostics[:n] = n
    diagnostics[:p] = p

    diagnostics
end

CSV.write("basis_pursuit.csv", results)
