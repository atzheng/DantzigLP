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


function group_dantzig_initializer(X, y, g, λ; eps=1e-8, maxiter=3e8)
    n = length(y)
    λ_n = λ / n
    results = rcopy(R"gglasso::gglasso($X, $y, group=$g, loss='ls', lambda=$λ_n, eps=$eps, maxit=$maxiter, intercept=FALSE)")
    return sparse(vec(results[:beta]))
end


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
