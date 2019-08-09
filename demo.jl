# Usage examples for several of the common problems in the DantzigLP framework
using DantzigLP

# Dantzig Selector
# ==============================================================================
X, y, actual = regression_example(100, 200, 0.2)
# Returns coefficient vector, model object and DataFrame with diagnostics.
β, model, log_fast = dantzig_lp(X, y, 1)
# Compare against same problem with column / constraint generation disabled
β, model, log_slow = dantzig_lp(X, y, 1;
                                column_generation=false,
                                constraint_generation=false)

# Basis Pursuit
# ==============================================================================
X, y, actual = basis_pursuit_example(100, 200, 0.2)
β, model, log_fast = basis_pursuit(X, y)
β, model, log_fast = basis_pursuit(X, y;
                                   column_generation=false,
                                   constraint_generation=false)

# Fused Dantzig Regression
# ==============================================================================
noisy, clean, a = DantzigLP.trend_filtering_example(100, 0, 20)
β, model = dantzig_tf(y, 0.1, 20)
β, model = dantzig_tf(y, 0.1, 20;
                      column_generation=false,
                      constraint_generation=false)
