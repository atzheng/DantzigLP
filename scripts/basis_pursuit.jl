using DantzigLP, CSV
include("utils.jl")

config = parse_config(ARGS[1])

# Warmup
X, y, β = DantzigLP.basis_pursuit_example(100, 500, 0.1)
DantzigLP.baseline_basis_pursuit(X, y; timeout=300)
DantzigLP.basis_pursuit(X, y; max_columns=40, verbose=false)

# Benchmarks
s = 0.2 * config[:n] / config[:p]
n = config[:n]
p = config[:p]

X, y, β = DantzigLP.basis_pursuit_example(n, p, s)

if p < 100000
    bl_time = @elapsed model, βdantzig = DantzigLP.baseline_basis_pursuit(
        X, y; timeout=600, Method=1)
else
    bl_time = nothing
end

cg_time = @elapsed βbp, model, diagnostics = DantzigLP.basis_pursuit(
    X, y; max_columns=40, verbose=false, solver_params=Dict(:Method => 1))

instance_id = hash(ARGS[1])

diagnostics[:baseline_secs] = bl_time
diagnostics[:total_secs] = cg_time
diagnostics[:instance_id] = repr(instance_id)
diagnostics[:n] = n
diagnostics[:p] = p

CSV.write(@sprintf("basis_pursuit/results/%s.csv", ARGS[1]), diagnostics)
