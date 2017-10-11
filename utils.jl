"""
Generate test regression problems. TODO Add D matrix somehow
"""
function generate_regression_example(n, p, density, SNR=10, binary=false)
    if binary
        X = round(rand(n, p))
    else
        X_unnorm = randn(n, p)
        X = mapslices(x -> x / norm(x, 2), X_unnorm, [1])
    end

    true_coeffs = round.(Int, rand(p) .<= density) .* rand(p) .* 10
    y = X * true_coeffs + randn(n) * var(X * true_coeffs) / SNR

    return X, y, true_coeffs
end

