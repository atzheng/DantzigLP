module FusedDantzig
include("dantzig.jl")
using Lasso

function fused_dantzig(y, delta;
                       X = speye(length(y)),
                       initializer = nothing,
                       args...)
    n = length(y)
    if initializer == nothing
        if X == speye(length(y))
            initializer = fused_lasso_initializer
        else
            initializer = lasso_initializer
        end
    end

    fused_X = X * fusion_operator(n)
    dantzig_transform =
        speye(n) - (fused_X[:, 1] * fused_X[:, 1]') / sum(fused_X[:, 1] .^ 2)

    dantzig_X = dantzig_transform * fused_X[:, 2:end]
    dantzig_y = dantzig_transform * y

    model, coefs = DantzigLP.dantzig_lp(dantzig_y, dantzig_X, delta;
                                        initializer = initializer, args...)
    original_coefs = recover_original_coefs(fused_X, y, coefs)
    return model, original_coefs
end

function recover_original_coefs(fused_X, y, coefs)
    n = size(fused_X)[1]
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
    return sparse(inv_fusion_operator(n) * fit(FusedLasso, y, delta).β)
end

end
