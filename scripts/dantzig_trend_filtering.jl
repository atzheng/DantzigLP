using DantzigLP, CSV
include("utils.jl")

config = parse_config(ARGS[1])
k = get(config, :k, 0)

# Warmup
noisy, clean, α = DantzigLP.tf_example(200, k, 10)
λ = 10
DantzigLP.dantzig_tf(noisy, λ, k;
                     column_generation=true, constraint_generation=true)
DantzigLP.baseline_dantzig_tf(noisy, λ, k)

# Generate problem
instance_id = hash(config)

noisy, clean, α =
    DantzigLP.trend_filtering_example(config[:n], k, config[:knots])
σ = sqrt(var(noisy - clean))
λ = σ * sqrt(2 * log(config[:n]))

# Run instance
params = Dict(:Method => 1)

bl_time = @elapsed βbl, model, bl_diagnostics =
    DantzigLP.baseline_dantzig_tf(
        noisy, λ, k; solver_params=params, verbose=true, return_α=true)

cg_time = @elapsed βbp, model, cg_diagnostics =
    DantzigLP.dantzig_tf(noisy, λ, k;
                         column_generation=true,
                         solver_params=params,
                         constraint_generation=true,
                         verbose=true, return_α=true)

cg_diagnostics[:baseline_secs] = bl_time
cg_diagnostics[:baseline_construction_secs] =
    bl_diagnostics[:construction_seconds]
cg_diagnostics[:baseline_gurobi_secs] =
    bl_diagnostics[:gurobi_seconds]
cg_diagnostics[:ccg_secs] = cg_time
cg_diagnostics[:instance_id] = repr(instance_id)
cg_diagnostics[:n] = config[:n]
cg_diagnostics[:knots] = config[:knots]
cg_diagnostics[:k] = config[:k]
cg_diagnostics[:L0] = norm(βbl, 0)

CSV.write(@sprintf("dantzig_tfk/results/%s.csv", ARGS[1]), cg_diagnostics)
