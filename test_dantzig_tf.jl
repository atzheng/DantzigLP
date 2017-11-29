function proj_op(A)
    return A * inv(full(A'A)) * A'
end

function test_FMD_getindex(n, k)
    normalizer = factorial(k) / (n ^ k)
    invdiff = tibs_invdiff(n, k)
    proj = proj_op()
    a = (eye(n) - proj) * invdiff[:, (k + 1):end]
    z = FusedDantzig.FusedDantzigMatrix(n, k)
    return isapprox(a, z)
end

z = FusedDantzig.FusedDantzigMatrix(10, k)
z[:, [3, 4, 6]]

@test test_FMD_getindex(10, 2)
@test test_FMD_getindex(100, 2)
@test test_FMD_getindex(10, 5)
@test test_FMD_getindex(10, 5)


x = [1, 3, 6, 2]
sx = sort(x)
shift = FusedDantzig.shift
accumulate((vec, i) -> shift(vec, i), [1,2,3,4,5,6], sx - shift(sx, 1))


# These should all agree
model1, coefs1 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 1, verbose = true)
model2, coefs2 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 1, initializer_fn = DantzigLP.lasso_initializer,
    verbose = true)
model3, coefs3 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 1, reduced_cost_fn = DantzigLP.get_reduced_costs,
    verbose = true)
model4, coefs4 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 1, lazy_X = true, verbose = true)


Xtil = FusedDantzig.FusedDantzigMatrix(n, 2);
max_delta = maximum(abs.([dot(Xtil[:, j], y) for j in 1:(n - 2)]))
model1, coefs1 = FusedDantzig.dantzig_trend_filtering(y, max_delta * 0.3, 2)
model2, coefs2 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 2, initializer_fn = DantzigLP.lasso_initializer)
model3, coefs3 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 2, reduced_cost_fn = DantzigLP.get_reduced_costs)
model4, coefs4 = FusedDantzig.dantzig_trend_filtering(
    y, max_delta * 0.3, 2, lazy_X = true)
