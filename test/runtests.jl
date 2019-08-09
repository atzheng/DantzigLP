using Base.Test, JuMP

using DantzigLP


# Test Dantzig Selector
# ------------------------------------------------------------------------------
X, y, actual = regression_example(100, 200, 0.2)
β, model, log_fast = dantzig_lp(X, y, 1)
β_colgen, _, _ = dantzig_lp(X, y, 1; constraint_generation=false)
β_congen, _, _ = dantzig_lp(X, y, 1; column_generation=false)
bl_model, β_baseline = baseline_dantzig(X, y, 1)
@test isapprox(β, β_baseline)
@test isapprox(β, β_colgen)
@test isapprox(β, β_congen)
@test isapprox(getobjectivevalue(model.gurobi_model),
               getobjectivevalue(bl_model))

# Test Basis Pursuit
# ------------------------------------------------------------------------------
X, y, actual = basis_pursuit_example(100, 200, 0.2)
β, model, log_fast = basis_pursuit(X, y; max_columns=40, verbose=false)
bl_model, β_baseline = baseline_basis_pursuit(X, y)
@test isapprox(β, β_baseline)
@test isapprox(getobjectivevalue(model.gurobi_model),
               getobjectivevalue(bl_model))

# Test Dantzig trend filtering
# ------------------------------------------------------------------------------
noisy, clean, a = DantzigLP.trend_filtering_example(100, 0, 20)
β_baseline, bl_model = DantzigLP.baseline_dantzig_tf(
    noisy, 1, 0; timeout=300, verbose=true)
β, model = dantzig_tf(noisy, 1, 0;
                      column_generation=true,
                      constraint_generation=true)
# Solutions are sometimes different, but objective values should be the same.
@test isapprox(getobjectivevalue(model.gurobi_model),
               getobjectivevalue(bl_model.gurobi_model))
