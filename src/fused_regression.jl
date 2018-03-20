using Lasso


function fused_regression(X, y, λ; args...)
    problem = FusedRegressionProblem(X, y, λ)
    L = norm(X, 2) .^ 2
    return FISTA(problem, L; args...)
end


type FusedRegressionProblem
    X
    y
    λ
end


function fista_objective(problem::FusedRegressionProblem, β)
    @unpack X, y, λ = problem
    n, p = size(X)
    return norm(y - X * β, 2) ^ 2 + λ * norm(difference_operator(p, 0) * β, 1)
end


function prox(problem::FusedRegressionProblem, β, L)
    @unpack X, y, λ = problem
    ∇f = - (X') * (y - X * β)
    β_new = fit(FusedLasso, β .- (1 / L) * ∇f, λ).β
end


function fused_regression_example(n, p, knots; SNR=10)
    X, _, _ = regression_example(n, p, 0.1)
    _, β, α = tf_example(p, 0, knots)

    signal = X * β
    y = add_gaussian_noise(signal, SNR)

    return X, y, β, α
end


function fused_regression_initializer(X, y, λ; args...)
    n, p = size(X)
    β = fused_regression(X, y, λ; args...)
    α_B = difference_operator(p, 1) * β
    return α_B
end


function dantzig_fused_regression(X, y, λ; args...)
    projected_X, projected_y = project_data(X, y)
    model = BasicDantzigModel(projected_X, projected_y)
    initial_soln, initializer_seconds =
        @timed fused_regression_initializer(X, y, λ)
    α_B, diagnostics = solve_dantzig_lp!(model, λ, initial_soln; args...)
    α, β = recover_coefs(X, y, α_B)
    diagnostics[:initializer_seconds] = initializer_seconds
    return β, model, diagnostics, α
end


function project_data(X, y)
    n, p = size(X)
    fused_X = X * LowerTriangular(ones(p, p))
    z = fused_X[:, 1]

    projection(x) =  z * (z'x) / (z'z)
    fused_projection(x) = x - projection(x)

    projected_y = fused_projection(y)
    projected_X = fused_projection(fused_X[:, 2:end])
    return projected_X, projected_y
end


function recover_coefs(X, y, α_B)
    n, p = size(X)
    fused_X = X * LowerTriangular(ones(p, p))
    z = fused_X[:, 1]

    residuals = y - fused_X[:, 2:end] * α_B
    α_a = (z'residuals) / z'z
    α = vcat(α_a, α_B)
    β = LowerTriangular(ones(p, p)) * full(α)
    return sparse(α), β
end


function baseline_dantzig_fused_regression(X, y, λ; args...)
    projected_X, projected_y = project_data(X, y)
    model, α_B = baseline_dantzig(projected_X, projected_y, λ; args...)
    α, β = recover_coefs(X, y, α_B)
    return β, model, α
end
