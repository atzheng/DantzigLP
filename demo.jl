# Usage examples for several of the common problems in the DantzigLP framework
using DantzigLP

# Dantzig Selector
# ==============================================================================
# generates dataset 
# y_{100} is response, X_{100 \times 200}=model matrix, 
# actual=true regression coefficients with 0.2 proportion of nonzero entries
X, y, actual = regression_example(100, 200, 0.2)

# Returns coefficient vector, model object and DataFrame with diagnostics.
# solves: min \| \beta \|_1  s.t. \|X'(y-X\beta)\|_\infty \leq \lambda
# here, lambda=1 (and X is assumed to have unit L2-norm features)
# column and constraint generation enabled.
β, model, log_fast = dantzig_lp(X, y, 1)

# Compare against same problem with column / constraint generation disabled
β, model, log_slow = dantzig_lp(X, y, 1;
                                column_generation=false,
                                constraint_generation=false)

# Basis Pursuit
# ==============================================================================
#Similar to above, with y=X*actual
X, y, actual = basis_pursuit_example(100, 200, 0.2)

# run basis pursuit with column generation enabled
β, model, log_fast = basis_pursuit(X, y)

# run basis pursuit with column generation disabled 
# (here constraint generation is vacuous) [Andy, can we remove constraint_generation= option?]
β, model, log_fast = basis_pursuit(X, y;
                                   column_generation=false,
                                   constraint_generation=false)

# Fused Dantzig Regression
# ==============================================================================
# generates a trend filtering example [for fused Lasso]
# signal of length=100, number of jumps=20
noisy, clean, a = DantzigLP.trend_filtering_example(100, 0, 20)

# Run Dantzig Selector version of fused lasso
# y=signal,  0.1=value of regularization parameter, k=0 specifies fused lasso.
# the following used both column and constraint generation
β, model = dantzig_tf(y, 0.1, 20)

# the following disables both column and constraint generation
β, model = dantzig_tf(y, 0.1, 20;
                      column_generation=false,
                      constraint_generation=false)

