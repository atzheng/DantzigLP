type BasisPursuitModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    size             # (n, p) tuple
    pos_betas        # Postive Beta variables already in the model (mutable)
    pos_beta_indices # Indices corresponding to entries of pos_betas (mutable)
    neg_betas        # Negative Beta variables already in the model (mutable)
    neg_beta_indices # Indices corresponding to entries of neg_betas (mutable)
    residual_constrs # Constraints r = y - X * Beta
    X                # Data matrix X
end


function BasisPursuitModel(X, y)
    n, p = size(X)
    model = Model()

    @constraints model begin
        residual_constrs, y .== 0
    end

    @objective(model, Min, 0)

    return BasisPursuitModel(model, (n, p), [], [], [], [], residual_constrs, X)
end


function initialize_constraints!(model::BasisPursuitModel, λ)
end


function basis_pursuit_initializer(X, y)
    n, p = size(X)
    lasso = fit(LassoPath, X, y, maxncoef=p, standardize=false, intercept=false)
    lasso_p = [norm(lasso.coefs[:, i], 0) for i in 1:length(lasso.λ)]
    coef_idx = minimum(find(lasso_p .≥ n))
    return sparse(lasso.coefs[:, coef_idx])
end


function basis_pursuit(X, y; args...)
    args_dict = Dict(args)
    verbose = get(args_dict, :verbose, false)
    vinfo(msg) = verbose_info(verbose, msg)

    # Don't allow constraint generation
    push!(args, (:constraint_generation, false))
    construction_secs = @elapsed model = BasisPursuitModel(X, y)
    vinfo(@sprintf("Initial model completed in %.2f seconds.",
                   construction_secs))

    initializer_secs = @elapsed initial_soln = basis_pursuit_initializer(X, y)
    vinfo(@sprintf("Initialization finished in %.2f seconds. L₀(β₀) = %d.",
                   initializer_secs, norm(initial_soln, 0)))

    β, diagnostics = solve_dantzig_lp!(model, 0, initial_soln; args...)
    return β, model, diagnostics
end


function baseline_basis_pursuit(X, y; verbose=true)
    n, p = size(X)
    model = Model(solver = GurobiSolver(OutputFlag = ifelse(verbose, 1, 0)))

    @variables model begin
        β⁺[1:p] ≥ 0
        β⁻[1:p] ≥ 0
    end

    @constraints model begin
        y - X * (β⁺ - β⁻) .== 0
    end

    @objective(model, Min, sum(β⁺ .+ β⁻))

    solve(model)
    return model, sparse(getvalue(β⁺) - getvalue(β⁻))
end
