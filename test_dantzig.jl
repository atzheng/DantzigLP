using Base.Test

# Test setup
p = 20000
n = 1000
X, y, actual = generate_regression_example(n, p, 0.2)
delta = maximum(abs(X'y)) * 0.8

baseline_model, baseline_soln = DantzigLP.dantzig_lp(y, X, delta, :simple)
gen_model, gen_soln, _ = DantzigLP.colgen_dantzig(
    y, X, delta, column_generation = false, constraint_generation = false)
colgen_model, colgen_soln, _ = DantzigLP.colgen_dantzig(
    y, X, delta, column_generation = true, constraint_generation = false)
congen_model, congen_soln, lasso_soln = DantzigLP.colgen_dantzig(
    y, X, delta, column_generation = false, constraint_generation = true)
colcongen_model, colcongen_soln, lasso_soln = DantzigLP.colgen_dantzig(
    y, X, delta, column_generation = true, constraint_generation = true)

results = []
max_delta = maximum(abs(X'y))
for alpha in 0.1:0.1:0.9
    colcongen_model, colcongen_soln, lasso_soln = DantzigLP.colgen_dantzig(
        y, X, max_delta * alpha,
        column_generation = true, constraint_generation = true)
    push!(results, (length(intersect(colcongen_soln.nzind, lasso_soln.nzind)),
                    length(colcongen_soln.nzind)))
end

# Test that all methods return the same solution
@test baseline_soln == gen_soln
@test baseline_soln == colgen_soln
@test baseline_soln == congen_soln
@test baseline_soln == colcongen_soln

# Test that reduced cost generation is correct
vec(trunc(DantzigLP.get_reduced_costs(baseline_model, X), 12)) ==
    trunc(baseline_model.redCosts[1:p], 12)


lasso_fit = fit(LassoPath, X, y)
lasso_fit.coefs[: , 8]
lasso_l1_vals = mapslices(x -> sum(abs(x)), lasso_fit.coefs, [1])
lasso_lambdas = lasso_fit.λ

delta = 15
z = DantzigLP.initialize_lasso_soln(X, y, delta, 20)
norm(z, 1) - delta


for i in 1:10
    @show i
end


