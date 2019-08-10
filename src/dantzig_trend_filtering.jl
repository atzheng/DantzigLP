type DantzigTFModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    size             # (n, p) tuple
    pos_betas        # JuMP variables for α⁺ already in the model
    pos_beta_indices # Indices corresponding to entries of pos_betas
    neg_betas        # JuMP variables for α⁻ already in the model
    neg_beta_indices # Indices corresponding to entries of neg_betas
    linf_pos_constrs # Constraints X'r <= λ
    linf_neg_constrs # Constraints X'r >= -λ
    α_constrs        # Constraints on the values of α⁺ and α⁻
    β                # JuMP Variables representing the β values
    ∇                # JuMP Variables representing the gradient value
    k                # Trend Filtering order
    λ                # TODO Currently unused; What to do with λ path?
end


"""
Runs `k`-order Dantzig trend filtering on a noisy signal `y`, with
regularization coefficient `λ`. `k=0` corresponds to the fused Dantzig selector.

Only solves for a single `λ` value; `λ` path not yet implemented. Accepts the
same keyword arguments as `dantzig_lp`.

# Additional arguments:
- return_α: return the recovered (sparse) knots, instead of the recovered signal.
"""
function dantzig_tf(y, λ, k; return_α=false, lasso_tol=1e-9, rounding_tol=1e-8,
                    initial_soln=nothing, args...)
    args_dict = Dict(args)
    colgen = get(args_dict, :column_generation, true)
    congen = get(args_dict, :constraint_generation, true)
    verbose = get(args_dict, :verbose, false)
    vinfo(msg) = verbose_info(verbose, msg)

    if !colgen && !congen
        return baseline_dantzig_tf(y, λ, k; return_α = return_α, args...)
    end

    n = length(y)

    vinfo("Constructing initial model...")
    construction_secs = @elapsed model = DantzigTFModel(y, λ, k)
    vinfo(@sprintf("Initial model completed in %.2f seconds.",
                   construction_secs))

    if initial_soln == nothing
        vinfo("Starting initializer...")
        initializer_secs = @elapsed initial_soln = trend_filtering_initializer(
            y, λ, k; rounding_tol = rounding_tol, tol = lasso_tol)
        vinfo(@sprintf("Initialization finished in %.2f seconds.",
                       initializer_secs))
    end

    α, diagnostics = solve_dantzig_lp!(model, λ, initial_soln; args...)

    diagnostics[:construction_seconds] += construction_secs
    diagnostics[:initializer_seconds] = initializer_secs

    if return_α
        soln = α
    else
        soln = getvalue(model.β)
    end
    return soln, model, diagnostics
end


function DantzigTFModel(y, λ, k)
    n = length(y)
    model = Model()

    @variables model begin
        αⁱ[1:n, 0:k]
        ∇ⁱ[1:n, 0:k]
    end

    @constraints model begin
        α_constrs, M⁻¹(n, k) * αⁱ[:, k] .== 0
        M⁻¹(n, 0)'∇ⁱ[:, 0] .== y .- αⁱ[:, 0]

        ∇ⁱ[1:k + 1, k] .== 0
    end

    for i in 0:(k - 1)
        @constraint(model, M⁻¹(n, i) * αⁱ[:, i] .== αⁱ[:, i + 1])
        @constraint(model, M⁻¹(n, i + 1)'∇ⁱ[:, i + 1] .== ∇ⁱ[:, i])
    end

    obj = @objective(model, Min, 0)

    return DantzigTFModel(model, (n, n), [], [], [], [], [], [],
                          α_constrs, αⁱ[:, 0], ∇ⁱ[:, k], k, λ)
end


function add_Xtr_constr!(model::DantzigTFModel, λ::Number,
                         idx::Integer, sign::Integer)
    if sign == -1
        new_constr = @constraint(model.gurobi_model, model.∇[idx] >= -λ)
        push!(model.linf_neg_constrs, new_constr)
    elseif sign == 1
        new_constr = @constraint(model.gurobi_model, model.∇[idx] <= λ)
        push!(model.linf_pos_constrs, new_constr)
    end
    return new_constr
end


function add_beta!(model::DantzigTFModel, idx::Integer, sgn::Integer)
    if sgn > 0
        beta_indices = model.pos_beta_indices
        betas = model.pos_betas
    else
        beta_indices = model.neg_beta_indices
        betas = model.neg_betas
    end

    if idx in beta_indices
        return nothing
    else
        new_var = @variable(model.gurobi_model,
                            objective = ifelse(idx > model.k + 1, 1, 0),
                            inconstraints = [model.α_constrs[idx]],
                            coefficients = [-sgn * 1.0],
                            lowerbound = 0)
        push!(beta_indices, idx)
        push!(betas, new_var)
        return new_var
    end
end


function get_reduced_costs(model::DantzigTFModel)
    # First n constraints correspond to the α constraints.
    n, p = model.size
    duals = model.gurobi_model.linconstrDuals[1:n]
    return (1 .+ duals, 1 .- duals)
end


function get_constraint_violations(model::DantzigTFModel)
    return getvalue(model.∇)
end


""" Uses the Lasso.jl implementation of the 1D fused lasso DP algorithm."""
function trend_filtering_initializer(y, λ, k; rounding_tol=1e-8, args...)
    n = length(y)
    if k == 0
        β = StatsBase.fit(FusedLasso, y, λ).β
        α_B = sparse(β[2:end] - β[1:end - 1])
    else
        raw_α_B = StatsBase.fit(TrendFilter, y, k, λ; args...).Dkp1β
        α_B = round_small(raw_α_B, rounding_tol)
    end

    results = spzeros(n)
    results[1:k + 1] = 1
    results[k + 2:end] = α_B
    return results
end

function round_small(X, tol)
    Y = spzeros(size(X)...)
    Y[abs.(X) .>= tol] = X[abs.(X) .>= tol]
    return Y
end


function baseline_dantzig_tf(y, λ, k;
                             verbose = false, return_α = false,
                             timeout = Inf, tol = 1e-6, exact = true,
                             solver_params = Dict(), args...)
    start_time = time_ns()

    n = length(y)
    vinfo(msg) = verbose_info(verbose, msg)

    vinfo("Constructing model...")
    default_params = Dict([(:Method, ifelse(exact, -1, 2)),
                           (:Crossover, ifelse(exact, -1, 0)),
                           (:BarConvTol, ifelse(exact, 1e-8, tol))])
    params_w_defaults = merge(default_params, solver_params)
    print(params_w_defaults)
    solver = construct_solver(verbose = verbose,
                              tol = tol,
                              timeout = timeout;
                              params_w_defaults...)

    model = Model(solver=solver)

    @variables model begin
        α⁺[1:n] >= 0
        α⁻[1:n] >= 0
        αⁱ[1:n, 0:k]
        ∇ⁱ[1:n, 0:k]
    end

    @constraints model begin
        α_constrs, M⁻¹(n, k) * αⁱ[:, k] .== α⁺ - α⁻
        M⁻¹(n, 0)'∇ⁱ[:, 0] .== y .- αⁱ[:, 0]

        ∇ⁱ[1:k + 1, k] .== 0
        ∇ⁱ[k + 2:n, k] .≤ λ
        ∇ⁱ[k + 2:n, k] .≥ -λ
    end

    for i in 0:(k - 1)
        @constraint(model, M⁻¹(n, i) * αⁱ[:, i] .== αⁱ[:, i + 1])
        @constraint(model, M⁻¹(n, i + 1)'∇ⁱ[:, i + 1] .== ∇ⁱ[:, i])
    end

    obj = @objective(model, Min, sum(α⁺[k + 2:end] .+ α⁻[k + 2:end]))
    construction_secs = (time_ns() - start_time) / 1e9


    vinfo("Solving model...")
    solve_start = time_ns()
    solve_status = solve(model)
    solve_secs = (time_ns() - solve_start) / 1e9

    diagnostics = DataFrame(
        delta = λ,
        construction_seconds = construction_secs,
        gurobi_seconds = solve_secs
    )

    if !(solve_status in [:Optimal, :Suboptimal])
        warn("Non-optimal solution")
        return solve_status, model, diagnostics
    end

    dtf_model = DantzigTFModel(model, (n, n),
                               α⁺, collect(1:n), α⁻, collect(1:n),
                               [], [], α_constrs, αⁱ[:, 0], ∇ⁱ[:, k], k, λ)

    if return_α
        α_values = sparse(getvalue(α⁺) - getvalue(α⁻))
        return α_values, dtf_model, diagnostics
    else
        β = getvalue(αⁱ[:, 0])
        return β, dtf_model, diagnostics
    end
end


function M⁻¹(n, k)
    return [speye(k + 1)        zeros(k + 1, n - k - 1)
            zeros(n - k - 1, k) difference_operator(n - k, 1)]
end


function difference_operator(n :: Integer, k :: Integer)
    if k == 0
        return speye(n, n)
    elseif k == 1
        operator = spzeros(n - 1, n)
        for i in 1:(n - 1)
            operator[i, i] = -1
            operator[i, i + 1] = 1
        end
        return operator
    else
        return difference_operator(n - k + 1, 1) *
            difference_operator(n, k - 1)
    end
end
