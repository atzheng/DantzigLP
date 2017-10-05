using StatsBase, Gadfly, DataFrames
using Lasso, GLMNet


include("regression.jl")


delta = 500
m = 1000
n = 100
X, y, actual = generate_regression_example(n, m, 0.2)
Y=y

X
y
actual
(y - X * actual) ./ y



path = glmnet(X, y)
# real_lasso_soln = fit(LassoPath, X, y, λ = [delta], maxncoef = m).coefs
real_lasso_soln = fit(LassoPath, X, y, maxncoef = m)
rm_lasso_soln = sparse(vec(DantzigLP.Lasso_soln_delta(y, X, delta)))
bd_model, bd_fit = DantzigLP.dantzig_lp(y, X, delta, :simple)


bl_model, bl_fit = baseline_regression(y, X)

delta = 1000
cg_model, cg_fit, lasso_fit = DantzigLP.colgen_dantzig(
    y, X, delta, column_generation = true, constraint_generation = false)

bd_fit
cg_fit



lambda = 6
D = speye(m)

imodel = internalmodel(bd_model)
p = MathProgBase.getconstrduals(imodel)
bd_model.linconstrDuals

x = DantzigLP.generate_column(bd_model, XtX, 1:m, vcat(1:m, 1:m))
x = DantzigLP.generate_column(bd_model, XtX, [], vcat(1:m, 1:m))




A = [1 3 0 4 1
     1 2 0 -3 1
     -1 -4 3 0 0]
b = [2 2 1]'
c = [2, 3, 3, 1, -2]

model = Model(solver = GurobiSolver())
x = @variable(model, [1:5], lowerbound = 0)
constrs = @constraint(model, A * x .== b)
obj = @objective(model, Min, dot(c, x))

solve(model)
getvalue(x)



