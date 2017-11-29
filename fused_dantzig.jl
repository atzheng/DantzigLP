module FusedDantzig
include("dantzig.jl")
include("utils.jl")
using Lasso
import Base.getindex, Base.size, Base.sort


type FusedDantzigMatrix <: AbstractMatrix{Number}
    n::Integer
    k::Integer
    # Stored values for fast computation
    cum_sums :: Vector  # Cumulative sums up to order k (n x k Array)
    U :: Array{Number, 2}  # From SVD of first k columns of H (n x k Array)
end

size(A :: FusedDantzigMatrix) = (A.n, A.n - A.k)


"""Typical constructor"""
function FusedDantzigMatrix(n::Int, k::Int)
    X1 = ones(n)
    # normalizer = factorial(k - 1) / (n ^ (k - 1))
    normalizer = 1
    cum_sums = cumsum_k(X1, k - 1) * normalizer

    Xa = hcat([shift(collect(1:n), j - 1)  for j in 1:k]...)
    # Xa = hcat([collect(1:n) .^ (j - 1) ./ n ^ (j - 1) for j in 1:k]...)
    U = svd(Xa)[1]
    return FusedDantzigMatrix(n, k, cum_sums, U)
end


function getindex(A::FusedDantzigMatrix, i::Colon, j::Integer)
    return vec(getindex(A, :, [j]))
end


function getindex(A::FusedDantzigMatrix, i::Colon, j::Vector)
    if A.k == 1
        upper = [- ones(jx) * (1 - jx / A.n) for jx in j]
        lower = [ones(A.n - jx) * (jx / A.n) for jx in j]
        Aj = hcat([vcat(u, l) for (u, l) in zip(upper, lower)]...)
    else
        if maximum(j) > A.n - A.k
            throw(BoundsError())
        else
            sorted_j = sort(j) .- 1 .+ A.k
            diffs = sorted_j - shift(sorted_j, 1)
            Xj = hcat(accumulate((vec, i) -> shift(vec, i), A.cum_sums, diffs)...)
            Aj = invdiff_matvecmult(A, Xj)
        end
    end
    return Aj
end


"""kth order cumulative sum"""
function cumsum_k(x::AbstractVector, k::Integer)
    if k == 0
        return x
    else
        return cumsum(cumsum_k(x, k - 1))
    end
end


function getindex(A :: FusedDantzigMatrix, i :: Integer, j :: Integer)
    if A.k == 1
        if i > A.n | j > (A.n - 1)
            throw(BoundsError())
        elseif i <= j
            return - (1 - j / A.n)
        else
            return j / A.n
        end
    else
        Xj = shift(A.cum_sums, A.k + j - 1)
        return (Xj - A.U * (A.U'Xj))[i]
    end
end


# TODO this is poorly named
function invdiff_matvecmult(A::FusedDantzigMatrix, x::Array)
    return x - A.U * (A.U'x)
end


# Dantzig trend filtering
# ==============================================================================
"""
Solves the Dantzig version of the 1D fused lasso. This version of the problem
has nice properties that enable O(n) search for minimum reduced cost and
analytic expressions for the X matrix.
"""
function fused_dantzig_1D(y, delta;
                          initializer_fn = fused_lasso_1D_initializer, args...)
    n = length(y)
    dantzig_X = FusedDantzigMatrix(n)
    dantzig_y = y .- sum(y .* 1 / n)

    model, coefs = DantzigLP.dantzig_lp(
        dantzig_y, dantzig_X, delta;
        initializer_fn = initializer_fn,
        reduced_cost_fn = get_reduced_costs,
        args...)

    original_coefs = recover_original_coefs(
        coefs, inv_difference_operator(n, 1), y, 1)
    return model, original_coefs
end


function dantzig_trend_filtering(y, delta, k;
                                 lazy_X = false,
                                 initializer_fn = tf_initializer(k),
                                 reduced_cost_fn = tf_reduced_cost_fn(k),
                                 args...)
    n = length(y)
    X_generator = FusedDantzigMatrix(n, k)
    dantzig_y = invdiff_matvecmult(X_generator, y)

    if lazy_X
        dantzig_X = X_generator
    else
        dantzig_X = X_generator[:, collect(1:(n - k))]
    end

    model, coefs = DantzigLP.dantzig_lp(dantzig_y, dantzig_X, delta;
                                        initializer_fn = initializer_fn,
                                        reduced_cost_fn = reduced_cost_fn,
                                        args...)
    original_coefs = recover_original_coefs(coefs, invdiff(X_generator), y, k)
    return model, original_coefs
end


# Function factories for trend filtering
function tf_reduced_cost_fn(k)
    if k == 1
        fn = fused_dantzig_reduced_costs
    else
        fn = DantzigLP.get_reduced_costs
    end
    return fn
end


function tf_initializer(k)
    initializer(X, y, delta) = trend_filtering_initializer(X, y, delta, k)
end



function baseline_dantzig_trend_filtering(y, delta, k;
                                          X = speye(length(y)), args...)
    n, p = size(X)
    S = inv_difference_operator(p, k)

    fused_X = X * S
    fusion = fusion_operator(fused_X, k)

    dantzig_X = fusion * fused_X[:, (k + 1):end]
    dantzig_y = fusion * y

    model, coefs = DantzigLP.dantzig_lp(dantzig_y, dantzig_X, delta; args...)
    original_coefs = recover_original_coefs(coefs, fused_X, y, k)
    return model, original_coefs
end


function recover_original_coefs(coefs_b, X, y, k)
    n = length(y)
    Xa = X[:, collect(1:k)]
    Xb = X[:, collect((k + 1):end)]

    X_generator = FusedDantzigMatrix(n, k)
    residuals = y - Xb * coefs_b
    coefs_a = inv(full(Xa'Xa)) * Xa'residuals

    return invdiff(X_generator) * vcat(coefs_a, coefs_b)
end


""" Uses the Lasso.jl implementation of the 1D fused lasso DP algorithm."""
function trend_filtering_initializer(X, y, delta, k)
    n = length(y)
    if k == 1  # TODO k numbering starrts from 1 here, as opposed to 0 in the original tf paper
        raw_coefs = fit(FusedLasso, y, delta).β
    else
        raw_coefs = fit(TrendFilter, y, k - 1, delta).β
    end
    return round_small(difference_operator(n, k) * raw_coefs, 1e-4)
end



"""
For k = 1 and X = I, we have a linear-time method for getting the reduced costs
based on the structure of the matrix in place of the data matrix.
"""
function fused_dantzig_reduced_costs(model)
    n, p = size(model.X)
    duals = model.gurobi_model.linconstrDuals[1:n]

    left_sum = 0
    right_sum = sum(duals)

    left_sums = spzeros(n)
    right_sums = spzeros(n)

    for i in 1:(length(n) - 1)
        left_sum += duals[i]
        right_sum -= duals[i]

        left_sums[i] = left_sum * (1 - i / n)
        right_sums[i] = right_sum * i / n
    end

    diffs = right_sums .- left_sums
    pos_costs = 1 .- diffs
    neg_costs = 1 .+ diffs

    return pos_costs, neg_costs
end


"""
Note: The original trend filtering paper has a different definition for this
matrix that has an explicit form for all k. However, this isn't useful unless
the fusion operator also has an explicit form for all k.
"""
function inv_difference_operator(n :: Integer, k :: Integer)
    vcat(speye(k, n), difference_operator(n, k)) |> full |> inv |> sparse
end


function fusion_operator(X :: AbstractMatrix, k :: Integer)
    n = size(X, 1)
    Xa = X[:, 1:k]
    return speye(n, n) - Xa * inv(full(Xa'Xa)) * Xa'
end


# Baselines
# ==============================================================================
# Inefficient implementations of difference and inverse difference operators,
# for testing and reference.
function difference_operator(n :: Integer, k :: Integer)
    if k == 0
        return speye(n, n)
    elseif k == 1
        operator = spzeros(n - 1, n)
        for i in 1:(n - 1)
            operator[i, i] = -1
            operator[i, i + 1] = 1
        end
        return operator
    else
        return difference_operator(n - k + 1, 1) *
            difference_operator(n, k - 1)
    end
end


function inv_difference_operator(n, k)
    diffops = [difference_operator(n, ki)[1, :] for ki in 0:k-1]
    first_k = hcat(diffops...)'
    rest = difference_operator(n, k)
    return vcat(first_k, rest) |> full |> inv
end


function round_small(x, tol)
    results = spzeros(length(x))
    for i in 1:length(x)
        if abs(x[i]) > tol
            results[i] = x[i]
        end
    end
    return results
end

function tibs_M(n, k)
    first_k = hcat(speye(k, k), spzeros(k, n - k))
    rest = hcat(spzeros(n - k, k), LowerTriangular(ones(n-k, n-k)))
    return vcat(first_k, rest)
end

function tibs_invdiff(n, k)
    normalizer = factorial(k - 1) / (n ^ (k - 1))
    return reduce(*, [tibs_M(n, i) for i in 0:k]) * normalizer |> full
end

function invdiff(A)
    # normalizer = factorial(A.k - 1) / (A.n ^ (A.k - 1))
    normalizer = 1
    @show(normalizer)
    # Xa = hcat([shift(collect(1:A.n), j - 1) .^ (j - 1) ./ A.n ^ (j - 1) for j in 1:A.k]...)
    Xa = hcat([shift(collect(1:A.n), j - 1)  for j in 1:A.k]...)
    cum_result = cumsum_k(ones(A.n), A.k - 1) .* normalizer
    Xb = hcat([sparse(shift(cum_result, i)) for i in A.k:(A.n - 1)]...)
    return hcat(Xa, Xb)
end

end
