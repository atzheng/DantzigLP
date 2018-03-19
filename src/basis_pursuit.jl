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
    # Do nothing
end


function basis_pursuit_initializer(X, y; λ_init=1, λ_n=30, λ_min=1e-7, args...)
    λ_max = norm(X'y, Inf)
    λ = λ_init
    is_solved = false
    βlasso = nothing

    while λ ≥ λ_min && !is_solved
        λs = log_range(λ_max, λ, λ_n)
        βlasso = basis_pursuit_initializer(X, y, λs; args...)
        Xb = X[:, βlasso.nzind]
        β = Xb \ y
        if Xb * β ≈ y
            is_solved = true
        end
        λ = λ / 2
    end

    if !is_solved
        error("λ_min reached; no feasible solution found.")
    end

    return βlasso
end


function basis_pursuit_initializer(X, y, λ; args...)
    n, p = size(X)
    lasso = fit(LassoPath, X, y, λ=λ / n,
                maxncoef=p, standardize=false, intercept=false; args...)
    lasso_p = [norm(lasso.coefs[:, i], 0) for i in 1:length(lasso.λ)]
    return sparse(lasso.coefs[:, end])
end


function log_range(minx, maxx, n)
    factor = exp((1 / n) * (log(maxx) - log(minx)))
    return minx .* (factor .^ collect(1:n))
end


function basis_pursuit(X, y; initializer_args=[], args...)

    args_dict = Dict(args)
    verbose = get(args_dict, :verbose, false)
    vinfo(msg) = verbose_info(verbose, msg)

    # Don't allow constraint generation
    push!(args, (:constraint_generation, false))
    construction_secs = @elapsed model = BasisPursuitModel(X, y)
    vinfo(@sprintf("Initial model completed in %.2f seconds.",
                   construction_secs))

    initializer_secs = @elapsed initial_soln =
        basis_pursuit_initializer(X, y; initializer_args...)

    vinfo(@sprintf(
        "Initialization finished in %.2f seconds. L₀(β₀) = %d.",
        initializer_secs, norm(initial_soln, 0)))

    β, diagnostics = solve_dantzig_lp!(model, 0, initial_soln; args...)

    diagnostics[:construction_secs] = construction_secs
    diagnostics[:initializer_secs] = initializer_secs

    return β, model, diagnostics
end


function baseline_basis_pursuit(X, y; args...)
    n, p = size(X)
    solver = construct_solver(; args...)
    model = Model(solver=solver)

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


function basis_pursuit_example(n, p, density)
    X, _, β = regression_example(n, p, density)
    y = X * β
    return X, y, sparse(β)
end
