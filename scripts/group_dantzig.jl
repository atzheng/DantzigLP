using DantzigLP, CSV
include("utils.jl")

config = parse_config(ARGS[1])

# Warmup
X, y, g, β = DantzigLP.group_dantzig_example(100, 200, 10, 0.1)
λ = norm((X') * (y - X * β), Inf)
group_dantzig(X, y, g, λ)
baseline_group_dantzig(X, y, g, λ)

# Generate problem
instance_id = hash(config)

s = 0.2 * config[:n] / config[:p]
X, y, g, β = DantzigLP.group_dantzig_example(
    config[:n], config[:p], config[:g], s)
λ = norm((X') * (y - X * β), Inf)

# Run instance
bl_time = @elapsed model, βdantzig =
    DantzigLP.baseline_group_dantzig(X, y, g, λ; Method = 1)
cg_time = @elapsed βbp, model, diagnostics =
    DantzigLP.group_dantzig(X, y, g, λ; verbose=true,
                            solver_params=Dict(:Method=>1))

diagnostics[:baseline_secs] = bl_time
diagnostics[:total_secs] = cg_time
diagnostics[:instance_id] = repr(instance_id)
diagnostics[:n] = config[:n]
diagnostics[:p] = config[:p]
diagnostics[:g] = config[:g]

CSV.write(@sprintf("group_dantzig/results/%s.csv", ARGS[1]), diagnostics)