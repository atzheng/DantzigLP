using StatsBase, Gadfly, DataFrames


m = 10000
n = 100
X, y, actual = generate_regression_example(10, m, 0.2)
X
y
actual
(y - X * actual) ./ y


bl_model, bl_fit = baseline_regression(y, X)
@profile cg_model, cg_fit = colgen_regression(y, X)

bl_fit - cg_fit


lambda = 6
D = speye(m)

imodel = internalmodel(cg_model)
p = MathProgBase.getconstrduals(imodel)
@time z = - transpose(p) * vcat(X, -D)




@time




rc = MathProgBase.getreducedcosts(imodel)[1:m]


new_var_type, new_var_idx = find_negative_reduced_cost(lambda, X, D, p)


get_regression_basis(6, X, eye(30), fit[1], error_constr, delta_constr)
