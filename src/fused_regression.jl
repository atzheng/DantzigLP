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


function baseline_dantzig_fused_regression(X, y, λ)
    n, p = size(X)
    fused_X = X * LowerTriangular(ones(p, p))
    z = fused_X[:, 1]

    projection(x) =  z * (z'x) / (z'z)
    fused_projection(x) = x - projection(x)

    projected_y = fused_projection(y)
    projected_X = fused_projection(fused_X[:, 2:end])
    model, α_B = baseline_dantzig(projected_X, projected_y, λ)

    residuals = y - fused_X[:, 2:end] * α_B
    α_a = (z'residuals) / z'z
    α = vcat(α_a, α_B)
    β = LowerTriangular(ones(p, p)) * full(α)
    return model, β, sparse(α)
end
