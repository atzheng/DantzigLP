using DantzigLP, CSV

config = parse_config(ARGS[1])

# Warmup
X, y, g, β = DantzigLP.group_dantzig_example(100, 200, 10, 0.1)
λ = norm((X') * (y - X * β), Inf)
group_dantzig(X, y, g, λ)
baseline_group_dantzig(X, y, g, λ)

# Generate problem
instance_id = hash(config)

s = 0.2 * config[:n] / config[:p]
X, y, g, β = DantzigLP.group_dantzig_example(config[:n], config[:p], config[:g], s)
λ = norm((X') * (y - X * β), Inf)

# Run instance
bl_time = @elapsed model, βdantzig = DantzigLP.baseline_group_dantzig(X, y, g, λ)
cg_time = @elapsed βbp, model, diagnostics =
    DantzigLP.group_dantzig(X, y, g, λ; verbose=true)

diagnostics[:baseline_secs] = bl_time
diagnostics[:total_secs] = cg_time
diagnostics[:instance_id] = repr(instance_id)
diagnostics[:n] = n
diagnostics[:p] = p

CSV.write(@sprintf("group_dantzig/results/%s.csv", ARGS[1]), diagnostics)
