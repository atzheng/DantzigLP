using DataFrames


"""
Generate test regression problems.
# Arguments
- `n`: Number of observations.
- `p`: Number of covariates.
- `density`: Proportion (between [0, 1]) of covariates with nonzero true
             coefficients. (poorly named)
- `SNR`: Signal to noise ratio. Typically between [10, 20].
- `binary`: Round X matrix to binary values.
- `rho`: Correlation coefficient.
- `sparsity`: Sparsity of the X matrix.

TODO Add D matrix somehow
"""
function generate_regression_example(n::Integer, p::Integer, density;
                                     SNR=10, binary=false, rho=0.0,
                                     sparsity=0.0)
    if binary
        X = sparse(round.(rand(n, p) * 0.6)) |> normalize_columns
    else
        # TODO Tried sprandn, which is an order of magnitude slower
        X = randn(n, p) |>
            A -> correlate(A, rho) |>
            A -> sparsify(A, sparsity) |>
            normalize_columns
    end

    true_coeffs = round.(Int, rand(p) .<= density) .* rand(p) .* 10
    y = X * true_coeffs + randn(n) * var(X * true_coeffs) / SNR

    return X, y, true_coeffs
end


"""
Set `pct` of X values to be zero.
TODO This is highly inefficient. Also maybe unnecessary
"""
function sparsify(X, pct)
    if pct == 0
        return X
    else
        n, p = size(X)
        spX = sparse(X)
        random_vals = rand(n, p)
        spX[random_vals .< pct] = 0
        return spX
    end
end


"""
Modify X such that all columns have a pairwise correlation of rho.
TODO Not sure if this preserves sparsity
"""
function correlate(X, rho)
    n = size(X)[1]

    if rho == 0.0
        return X
    else
        return broadcast(+, (1 - rho) * X / rho, randn(n, 1))
    end
end


"Normalize columns of a matrix by their L2 norm."
function normalize_columns(X)
    return mapslices(x -> x / norm(x, 2), X, [1])
end


"Convert a list of structs to a dataframe."
function struct2df(xs)
    fields = fieldnames(xs[1])
    return DataFrame(
        [[getfield(x, field) for x in xs] for field in fields],
        fields)
end
