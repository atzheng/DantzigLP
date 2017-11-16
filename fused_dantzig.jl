module FusedDantzig
include("dantzig.jl")
using Lasso
import Base.getindex, Base.size


type FusedDantzigMatrix <: AbstractMatrix{Number}
    n :: Int
end


size(A :: FusedDantzigMatrix) = (A.n, A.n - 1)


function getindex(A :: FusedDantzigMatrix, i :: Integer, j :: Integer)
    if i > A.n | j > (A.n - 1)
        throw(BoundsError())
    elseif i <= j
       return - (1 - j / A.n)
    else
        return j / A.n
    end
end


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


"""
TODO Add calls to the "Fast and Flexible ADMM Algorithms for TF" routine
for initialization.
"""
function dantzig_trend_filtering(y, delta, k; X = speye(length(y)), args...)
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


function recover_original_coefs(coefs_b, fused_X, y, k)
    n = length(y)
    Xa = fused_X[:, 1:k]
    Xb = fused_X[:, (k + 1):end]

    residuals = y - Xb * coefs_b
    coefs_a = inv(full(Xa'Xa)) * Xa'residuals

    return inv_difference_operator(n, k) * vcat(coefs_a, coefs_b)
end


""" Uses the Lasso.jl implementation of the 1D fused lasso DP algorithm."""
function fused_lasso_1D_initializer(X, y, delta)
    n = length(y)
    return sparse(difference_operator(n, 1) * fit(FusedLasso, y, delta).β)
end


"""
For X = I, we have a linear-time method for getting the reduced costs based
on the structure of the matrix in place of the data matrix.
"""
function get_reduced_costs(model)
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
TODO This is an inefficient implementation; there exists an explicit
form for entries of this matrix
"""
function difference_operator(n :: Integer, k :: Integer)
     if k == 1
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


end
