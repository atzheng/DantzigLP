using Base.Test

ε = 1e-6

function test_get_reduced_costs(model::DantzigModel)
    actual_costs = vcat(getdual(model.pos_betas),
                        getdual(model.neg_betas))
    test_pos_costs, test_neg_costs = get_reduced_costs(model)
    test_costs = vcat(test_pos_costs[model.pos_beta_indices],
                      test_neg_costs[model.neg_beta_indices])
    @test actual_costs ≈ test_costs atol=ε
end


function test_optimality(model::DantzigModel, β, baseline_β)
    @test objective(model, β) ≈ objective(model, baseline_β) atol=ε
end


function objective(model::DantzigModel, β)
    return norm(β, 1)
end


function objective(model::GroupDantzigModel, β)
    return sum([norm(βs, Inf) for (group, βs) in split_by(β, model.groups)])
end


function run_group_dantzig(; args...)
    # Generate problem
    n = 100
    p = 1000
    k = Int(ceil(p / 10))
    λ = 50
    X, y, g, true_β = group_dantzig_example(n, p, k, 0.2)

    # Get baseline
    baseline_β, _ = baseline_group_dantzig(X, y, g, λ)

    # Get model
    β, model, _ = group_dantzig(X, y, g, λ; args...)

    return (X, y, g, λ), model, β, baseline_β
end


function test_feasibility(model::GroupDantzigModel, problem, β)
    X, y, g, λ = problem
    Xtr = (X') * (y - X * β)
    group_norms = [norm(Xtr[find(g .== group)], 1) for group in 1:maximum(g)]
    @test norm(group_norms, Inf) - λ ≈ 0 atol=ε
end


# problems = [run_dantzig_lp, run_group_dantzig, run_dantzig_tf]
problems = [run_group_dantzig]
scenarios = [[(:column_generation, false), (:constraint_generation, false)],
             [(:column_generation, true), (:constraint_generation, false)],
             [(:column_generation, false), (:constraint_generation, true)],
             [(:column_generation, true), (:constraint_generation, true)]]


for problem in problems
    for scenario in scenarios
        problem, model, β, baseline_β = problem(; scenario...)

        @testset "Model Correctness" begin
            test_get_reduced_costs(model)
            test_feasibility(model, problem, β)
            test_optimality(model, β, baseline_β)
        end
    end
end
