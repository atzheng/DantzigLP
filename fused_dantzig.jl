module FusedDantzig
include("dantzig.jl")
using Lasso
import Base.getindex, Base.size


type FusedDantzigMatrix <: AbstractMatrix{Number}
    n :: Int
end


size(A :: FusedDantzigMatrix) = (A.n, A.n)


function getindex(A :: FusedDantzigMatrix, i :: Integer, j :: Integer)
    if i > A.n | j > A.n
        throw(BoundsError())
    elseif i <= j
        return - (1 - j / A.n)
    else
        return j / A.n
    end
end


function fused_dantzig(y, delta;
                       initializer_fn = fused_lasso_initializer, args...)
    n = length(y)
    dantzig_X = FusedDantzigMatrix(n)
    dantzig_y = y .- sum(y .* 1 / n)

    model, coefs = DantzigLP.dantzig_lp(
        dantzig_y, dantzig_X, delta;
        initializer_fn = initializer_fn,
        reduced_cost_fn = get_reduced_costs,
        args...)

    @show(coefs)
    original_coefs = recover_original_coefs(fusion_operator(n), y, coefs)
    return model, original_coefs
end


function generalized_fused_dantzig(y, delta;
                                   X = speye(length(y)),
                                   initializer_fn = nothing,
                                   args...)
    n = length(y)
    if initializer_fn == nothing
        if X == speye(length(y))
            initializer_fn = fused_lasso_initializer
        else
            initializer_fn = lasso_initializer
        end
    end

    fused_X = X * fusion_operator(n)
    dantzig_transform =
        speye(n) - (fused_X[:, 1] * fused_X[:, 1]') / sum(fused_X[:, 1] .^ 2)

    dantzig_X = dantzig_transform * fused_X[:, 2:end]
    dantzig_y = dantzig_transform * y

    model, coefs = DantzigLP.dantzig_lp(
        dantzig_y, dantzig_X, delta;
        initializer_fn = initializer_fn, args...)
    original_coefs = recover_original_coefs(fused_X, y, coefs)
    return model, original_coefs
end


function recover_original_coefs(fused_X, y, coefs)
    n = length(y)
    residuals = y - fused_X[:, 2:end] * coefs
    coef_1 = residuals'fused_X[:, 1] / sum(fused_X[:, 1] .^ 2)
    return fusion_operator(n) * vcat(coef_1, coefs)
end


function fusion_operator(n)
    return ones(n, n) |> LowerTriangular |> sparse
end


"""TODO: Very inefficient implementation for now."""
function inv_fusion_operator(n)
    return fusion_operator(n) |> full |> inv
end


"""
Uses the Lasso.jl implementation of the 1D fused lasso DP algorithm.
WARNING: Only valid for X = I.
"""
function fused_lasso_initializer(X, y, delta)
    n, p = size(X)
    return sparse(inv_fusion_operator(n) * fit(FusedLasso, y, delta).β)[2:end]
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

end
