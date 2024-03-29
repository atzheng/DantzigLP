using DataFrames, StatsBase, JuMP, Gurobi


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

"""
function regression_example(n::Integer, p::Integer, density;
                            SNR=10, binary=false, rho=0.0,
                            sparsity=0.0, standardize=false)
    if binary
        X = sparse(round.(rand(n, p) * 0.6)) |> normalize_columns
    else
        # Tried sprandn, which is an order of magnitude slower
        X = randn(n, p) |>
            A -> correlate(A, rho) |>
            A -> sparsify(A, sparsity) |>
            normalize_columns |>
            A -> applyif(standardize, standardize_columns, A)
    end

    true_coeffs = round.(Int, rand(p) .<= density) .* rand(p) .* 10
    y = X * true_coeffs + randn(n) * var(X * true_coeffs) / SNR |>
        xx -> applyif(standardize, standardize_column, xx)

    return X, y, true_coeffs
end


function standardize_columns(X)
    return hcat([standardize_column(X[:, j]) for j in 1:size(X, 2)]...)
end

function standardize_column(Xⱼ)
    return (Xⱼ - mean(Xⱼ)) / sqrt(var(Xⱼ))
end


function applyif(cond, fn, arg, args...)
    if cond
        return fn(arg, args...)
    else
        return arg
    end
end


function tibs_M(n::Number, k::Number)
    return tibs_M(Int(n), Int(k))
end

function tibs_M(n::Integer, k::Integer)
    first_k = hcat(speye(k, k), spzeros(k, n - k))
    rest = hcat(spzeros(n - k, k), LowerTriangular(ones(n-k, n-k)))
    return vcat(first_k, rest)
end


function trend_filtering_example(n::Integer, k::Integer, knots::Integer;
                                 SNR=10, continuous=false)
    knot_idx = vcat(sort(sample(1:(n - 1), Int(knots), replace=false)), [n])
    diffs = knot_idx .- shift(knot_idx, 1)
    piecewise_fns = [random_polynomial(diff, k) for diff in diffs]

    if continuous
        signal = reduce((x, y) -> vcat(x, x[end] .+ y .- y[1]), piecewise_fns)
    else
        signal = vcat(piecewise_fns...)
    end

    noisy = add_gaussian_noise(signal, SNR)
    return noisy, signal, knot_idx
end


function add_gaussian_noise(signal, SNR)
    var_noise = var(signal) / SNR
    noise = randn(length(signal)) * sqrt(var_noise)
    return signal + noise
end


function random_polynomial(n, k; min_k = 0)
    x_interval = rand(2) .* [-1, 1]
    coefs = randn(k - min_k + 1)
    step_size = (x_interval[2] - x_interval[1]) / n
    x = step_size .* collect(1:n) .+ x_interval[1]
    poly_xs = hcat([x.^i for i in min_k:k]...)
    return poly_xs * coefs
end


"""
Set `pct` of X values to be zero.
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


function shift(x::AbstractVector, k::Integer)
    y = circshift(x, k)
    y[1:min(end, k)] = 0
    return y
end

function verbose_info(verbose, msg)
    if verbose info(msg) end
end


function construct_solver(; verbose = false, tol = 1e-6, timeout = Inf, args...)
    default_params = Dict([(:OutputFlag, ifelse(verbose, 1, 0)),
                           (:TimeLimit, timeout),
                           (:FeasibilityTol, tol),
                           (:OptimalityTol, tol)])

    params_w_defaults = merge(default_params, Dict(args))
    solver = GurobiSolver(; params_w_defaults...)
    return solver
end


function projection_distance(x, y)
    return sum([minimum(abs.(yx .- x)) for yx in y])
end


"""
Merge the nonzero values of all vectors, with smaller indices of xs
taking precedence.
"""
function merge_vectors(xs...)
    if length(unique(map(length, xs))) > 1
        error("All vectors must have equal length.")
    end

    results = spzeros(length(xs[1]))
    nzinds = []
    for x in xs
        spx = sparse(x)
        new_nzind = setdiff(spx.nzind, nzinds)
        results[new_nzind] = spx[new_nzind]
        push!(nzinds, new_nzind)
    end

    return results
end


zip_collect(args...) = collect(zip(map(vec, args)...))
