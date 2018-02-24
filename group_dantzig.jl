using RCall

type GroupDantzigModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    size             # (n, p) tuple
    pos_betas        # JuMP variables for α⁺ already in the model
    pos_beta_indices # Indices corresponding to entries of pos_betas
    neg_betas        # JuMP variables for α⁻ already in the model
    neg_beta_indices # Indices corresponding to entries of neg_betas
    linf_pos_constrs # Constraints X'r <= λ
    linf_neg_constrs # Constraints X'r >= -λ
    α_constrs        # Constraints on the values of α⁺ and α⁻
    β                # JuMP Variables representing the β values
    ∇                # JuMP Variables representing the gradient value
    k                # Trend Filtering order
    λ                # TODO Currently unused; What to do with λ path?
end


function GroupDantzigModel(X, y, g, λ; args...)
end


function group_dantzig_example(n, p, k, density; SNR=10)
    X, _, _ = regression_example(n, p, density)
    groups = rand(1:k, p)

    n_eff_groups = Int(ceil(k * density))
    eff_groups = sortperm(rand(k))[1:n_eff_groups]

    is_eff_group = [g ∈ eff_groups for g in groups]
    true_coeffs = ifelse.(is_eff_group, rand(p) .* 10, 0)
    clean_y = X * true_coeffs
    noisy_y = clean_y + randn(n) * var(clean_y) / SNR
    return X, noisy_y, groups, true_coeffs
end


"Full LP implementation. For benchmarking and testing."
function baseline_group_dantzig(X, y, g, λ; args...)
    # TODO Assumes g has groups 1:k; shouldn't need this assumption
    n, p = size(X)
    k = length(unique(g))
    solver = construct_solver(; args...)
    model = Model(solver=solver)

    @variables model begin
        β⁺[1:p] ≥ 0
        β⁻[1:p] ≥ 0
        βg[1:k]
        ∇⁺[1:p] ≥ 0
        ∇⁻[1:p] ≥ 0
        r[1:n]
    end

    @constraints model begin
        r .== y - X * (β⁺ - β⁻)
        ∇⁺ .- ∇⁻ .== X'r
    end

    for i in 1:k
        group_jx = find(g .== i)
        @constraint(model, βg[i] .≥ β⁺[group_jx] + β⁻[group_jx])
        @constraint(model, sum(∇⁺ .+ ∇⁻) ≤ λ)
    end

    obj = @objective(model, Min, sum(βg))

    solve(model)
    β = sparse(getvalue(β⁺) .- getvalue(β⁻))
    return β, model, nothing
end


"FISTA solver for the L1/L∞ Group Lasso."
function group_lasso(X, y, g, λ; tol=1e-6, maxiter=1e6, L=nothing,
                     line_search=false, line_search_factor=0.5,
                     verbose=false)
    n, p = size(X)
    α = 0 # Acceleration parameter
    prox_β = spzeros(p)
    β = spzeros(p)

    L = ifelse(L == nothing, norm(X, 2) .^ 2, L)
    obj = group_lasso_objective(X, y, g, λ, β)
    vinfo(msg) = verbose_info(verbose, msg)

    is_converged = false
    for i in 1:maxiter
        Xᵀr = transpose(X) * (y - X * β)
        prox_β_new = prox(g, β, λ, Xᵀr, L)

        # Acceleration step
        α_new = (1 + sqrt(1 + 4 * α ^ 2)) / 2
        γ = (1 - α) / α_new
        β_new = (1 - γ) * prox_β_new + γ * prox_β

        obj_new = group_lasso_objective(X, y, g, λ, β_new)
        if abs(obj_new - obj) ≤ tol && i > 20
            # Case 1: Termination condition reached; return.
            is_converged = true
            break
        elseif obj_new > obj && line_search
            # Case 2: Gradient step makes objective worse; shrink step size and
            # retry.
            L = L / line_search_factor
            vinfo(@sprintf("Iteration %d: Reducing step_size to %.8f.",
                          i, L))
        else
            # Case 3: Gradient step improves objective; set new starting point
            # and continue.
            vinfo(@sprintf("Iteration %d: Objective %.2f", i, obj_new))
            α = α_new
            prox_β = prox_β_new
            β = β_new
            obj = obj_new
        end
    end

    if is_converged
        vinfo("Solution converged.")
    else
        vinfo("Max iterations reached.")
    end

    return sparse(β)
end


function group_lasso_objective(X, y, g, λ, β)
    group_norms = [norm(βg, Inf) for (_, βg) in split_by(β, g)]
    return 0.5 * norm(y - X * β, 2) ^ 2 + λ * norm(group_norms, 1)
end


function split_by(x, by)
    return [(by_val, x[find(by .== by_val)]) for by_val in unique(by)]
end


function vcat_by(xs, by)
    results = zeros(length(by))
    for (by_val, x) in xs
        results[find(by .== by_val)] = x
    end
    return results
end


function prox(g, β, λ, Xᵀr, L)
    a = β .+ (1 / L) * Xᵀr
    ag = split_by(a, g)

    βg_new = pmap(x -> (x[1], prox_group(x[2], λ, L)), ag)
    β_new = vcat_by(βg_new, g)
    return β_new
end

"""
Solve the proximal step for a group by projecting ag onto the L1 ball, which
yields an optimal dual solution.
"""
function prox_group(ag, λ, L)
    λLu = L1_projection(ag, z = λ / L)
    βg = ag - λLu
    return βg
end


function prox_group_gurobi(ag, λ, L)
    pg = length(ag)
    solver = GurobiSolver(OutputFlag=0)
    model = Model(solver=solver)

    @variables model begin
        β[1:pg]
        β∞ ≥ 0
    end

    @constraints model begin
        β[1:pg] .≤ β∞
        β[1:pg] .≥ -β∞
    end

    @objective(model, Min, (1 / (2 * L)) * sum((β - ag) .^ 2) + λ * β∞)

    solve(model)
    return getvalue(β)
end


function L1_projection(v; z=1.0)
    if norm(v, 1) ≤ z
        return v
    else
        return sign.(v) .* simplex_projection(abs.(v); z=z)
    end
end


"""
Projection onto the probabilistic simplex in O(n), from
https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
"""
function simplex_projection(v; z=1.0)
    n = length(v)
    U = collect(1:n)
    s = 0
    ρ = 0

    while length(U) > 0
        k = sample(U, 1)[1]
        G = filter(j -> v[j] ≥ v[k], U)
        L = filter(j -> v[j] < v[k], U)

        Δρ = length(G)
        Δs = sum([v[j] for j in G])

        if (s + Δs) - (ρ + Δρ) * v[k] < z
            s += Δs
            ρ += Δρ
            U = L
        else
            U = setdiff(G, [k])
        end
    end

    θ = (s - z) / ρ
    return max.(v .- θ, 0)
end
