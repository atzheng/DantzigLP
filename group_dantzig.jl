using ProgressMeter


type GroupDantzigModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    size             # (n, p) tuple
    # Variables
    pos_betas        # JuMP variables for α⁺ already in the model
    pos_beta_indices # Indices corresponding to entries of pos_betas
    neg_betas        # JuMP variables for α⁻ already in the model
    neg_beta_indices # Indices corresponding to entries of neg_betas
    residuals        # JuMP variables corresponding to residuals
    ∇⁺               # JuMP variables corresponding to X'r₊
    ∇⁻               # JuMP variables corresponding to X'r₋
    # Constraints
    residual_constrs # Constraints r = y - Xβ
    βg_constrs       # Constraints βg ≥ norm(β_i, ∞) ∀ i ∈ g, g ∈ G
    ∇_constr_indices # Indices corresponding to constraints ∇⁺ - ∇⁻ = X'r
    # Problem Data
    groups           # Vector mapping coefficients to groups
    X                # Data matrix X
end


function GroupDantzigModel(X, y, g, λ; args...)
    # TODO Should translate g into 1:k
    n, p = size(X)
    k = length(unique(g))
    model = Model()

    @variables model begin
        βg[1:k]
        ∇⁺[1:p] ≥ 0
        ∇⁻[1:p] ≥ 0
        r[1:n]
    end

    # Warning: the order in which we add these constraints affects how
    # get_reduced_costs works.
    @constraints model begin
        residual_constrs, r .== y
        βg_constrs[j=1:p], βg[g[j]] ≥ 0
    end

    for i in 1:k
        group_jx = find(g .== i)
        @constraint(model, sum(∇⁺[group_jx] .+ ∇⁻[group_jx]) ≤ λ)
    end

    obj = @objective(model, Min, sum(βg))

    return GroupDantzigModel(model, (n, p), [], [], [], [], r, ∇⁺, ∇⁻,
                             residual_constrs, βg_constrs, [], g, X)
end


function group_dantzig(X, y, g, λ; lasso_tol=1e-5, args...)
    args_dict = Dict(args)
    # Typically small number |G| of constraints; no congen for now.
    verbose = get(args_dict, :verbose, false)
    # args_dict[:constraint_generation] = false
    # # Column generation policy will be to add one group at a time.
    # args_dict[:max_columns] = 1
    vinfo(msg) = verbose_info(verbose, msg)

    vinfo("Constructing initial model...")
    construction_secs = @elapsed model = GroupDantzigModel(X, y, g, λ)
    vinfo(@sprintf("Initial model completed in %.2f seconds.",
                   construction_secs))

    vinfo("Starting initializer...")
    initializer_secs = @elapsed initial_soln =
        group_lasso(X, y, g, λ; tol=lasso_tol, verbose=verbose)
    vinfo(@sprintf("Initialization finished in %.2f seconds.",
                   initializer_secs))

    β, diagnostics = solve_dantzig_lp!(model, λ, initial_soln; args_dict...)
    return β, model, diagnostics
end


function initialize_constraints!(model::GroupDantzigModel, λ)
    k = maximum(model.groups)
    init_constrs = 1:k
    add_Xtr_constr!(model, λ, init_constrs, fill(1, k))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, k))
end


function initialize_constraints!(model::GroupDantzigModel, λ, initial_soln)
    init_constrs = unique(model.groups[initial_soln.nzind])
    n_nz = length(initial_soln.nzind)
    add_Xtr_constr!(model, λ, init_constrs, fill(1, n_nz))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, n_nz))
end


# Constraint Generation
# ==============================================================================
"""
Adds the L1 / L∞ constraint for a given group. Note that these are indexed
on groups, not on columns.
"""
function add_Xtr_constr!(model::GroupDantzigModel, λ::Number,
                         idx::Integer, sign::Integer)
    if idx in model.∇_constr_indices
        return nothing
    else
        new_constrs = [@constraint(model.gurobi_model,
                                   model.∇⁺[j] - model.∇⁻[j]
                                   == (model.X[:, j])'model.residuals)
                       for j in find(model.groups .== idx)]
        # @debug @sprintf("Added %d constraints for group %d.",
                        # length(new_constrs), idx)
        push!(model.∇_constr_indices, idx)
        return new_constrs
    end
end


function get_constraint_violations(model::GroupDantzigModel)
    Xᵀr = (model.X') * getvalue(model.residuals)
    return [norm(Xᵀr[find(model.groups .== g)], 1)
            for g in 1:maximum(model.groups)]
end


# Column Generation
# ==============================================================================
function add_beta!(model::GroupDantzigModel, idx::Integer, sgn::Number)
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
        new_var = @variable(
            model.gurobi_model,
            objective = 0,
            inconstraints = vcat(model.residual_constrs,
                                 [model.βg_constrs[idx]]),
            coefficients = vcat(sgn * full(model.X[:, idx]), [-1]),
            lowerbound = 0)
        # @debug @sprintf("Added %d β[%d].", sgn, idx)
        push!(beta_indices, idx)
        push!(betas, new_var)
        return new_var
    end
end


function generate_columns!(model::GroupDantzigModel, max_columns, tol)
    return invoke(generate_columns!, Tuple{DantzigModel, Any, Any},
                  model, max_columns, tol)
end

# """
# Generates all columns for a group at once. Adds the group corresponding to the
# column with the most negative reduced cost.
# """
# function generate_columns!(model::GroupDantzigModel, max_columns, tol)
#     new_columns = invoke(generate_columns!, Tuple{DantzigModel, Any, Any},
#                          model, 1, tol)

#     if length(new_columns) > 0
#         new_idx, new_sgn = new_columns[1]
#         new_group = model.groups[new_idx]

#         group_indices = find(model.groups .== new_group)
#         for idx in group_indices
#             add_beta!(model, idx, 1)
#             add_beta!(model, idx, -1)
#         end

#         group_size = length(group_indices)
#         return vcat(zip_collect(group_indices, fill(1, group_size)),
#                     zip_collect(group_indices, fill(-1, group_size)))
#     else
#         return new_columns
#     end
# end


function get_reduced_costs(model::GroupDantzigModel)
    n, p = model.size
    residual_duals = model.gurobi_model.linconstrDuals[1:n]
    βg_max_duals = model.gurobi_model.linconstrDuals[n + 1:n + p]

    pos_reduced_costs = - vec(residual_duals'model.X) .+ βg_max_duals
    neg_reduced_costs = vec(residual_duals'model.X) .+ βg_max_duals

    return (pos_reduced_costs, neg_reduced_costs)
end


# Group Lasso
# ==============================================================================
"FISTA solver for the L1/L∞ Group Lasso."
function group_lasso(X, y, g, λ; tol=1e-6, maxiter=1e6, L=nothing,
                     line_search=false, line_search_factor=0.5,
                     verbose=false)
    n, p = size(X)
    α = 0 # Acceleration parameter
    prox_β = spzeros(p)
    β = spzeros(p)

    L = ifelse(L == nothing, norm(X, 2) .^ 2, L)
    obj = group_lasso_objective(X, y, g, λ, β)
    vinfo(msg) = verbose_info(verbose, msg)

    is_converged = false

    if verbose
        progressbar = Progress(Int(maxiter))
    end

    for i in 1:maxiter
        Xᵀr = transpose(X) * (y - X * β)
        prox_β_new = prox(g, β, λ, Xᵀr, L)

        # Acceleration step
        α_new = (1 + sqrt(1 + 4 * α ^ 2)) / 2
        γ = (1 - α) / α_new
        β_new = (1 - γ) * prox_β_new + γ * prox_β

        obj_new = group_lasso_objective(X, y, g, λ, β_new)
        if abs(obj_new - obj) ≤ tol && i > 20
            # Case 1: Termination condition reached; return.
            is_converged = true
            break
        elseif obj_new > obj && line_search
            # Case 2: Gradient step makes objective worse; shrink step size and
            # retry.
            L = L / line_search_factor
            vinfo(@sprintf("Iteration %d: Reducing step_size to %.8f.",
                          i, L))
        else
            # Case 3: Gradient step improves objective; set new starting point
            # and continue.
            α = α_new
            prox_β = prox_β_new
            β = β_new
            obj = obj_new
        end

        if verbose
            ProgressMeter.next!(progressbar;
                                showvalues = [(:iter, i), (:obj, obj)])
        end
    end

    if is_converged
        vinfo("Solution converged.")
    else
        vinfo("Max iterations reached.")
    end

    return sparse(β)
end


function group_lasso_objective(X, y, g, λ, β)
    group_norms = [norm(βg, Inf) for (_, βg) in split_by(β, g)]
    return 0.5 * norm(y - X * β, 2) ^ 2 + λ * norm(group_norms, 1)
end


function split_by(x, by)
    return [(by_val, x[find(by .== by_val)]) for by_val in unique(by)]
end


function vcat_by(xs, by)
    results = zeros(length(by))
    for (by_val, x) in xs
        results[find(by .== by_val)] = x
    end
    return results
end


function prox(g, β, λ, Xᵀr, L)
    a = β .+ (1 / L) * Xᵀr
    ag = split_by(a, g)

    βg_new = pmap(x -> (x[1], prox_group(x[2], λ, L)), ag)
    β_new = vcat_by(βg_new, g)
    return β_new
end

"""
Solve the proximal step for a group by projecting ag onto the L1 ball, which
yields an optimal dual solution.
"""
function prox_group(ag, λ, L)
    λLu = L1_projection(ag, z = λ / L)
    βg = ag - λLu
    return βg
end


function prox_group_gurobi(ag, λ, L)
    pg = length(ag)
    solver = GurobiSolver(OutputFlag=0)
    model = Model(solver=solver)

    @variables model begin
        β[1:pg]
        β∞ ≥ 0
    end

    @constraints model begin
        β[1:pg] .≤ β∞
        β[1:pg] .≥ -β∞
    end

    @objective(model, Min, (1 / (2 * L)) * sum((β - ag) .^ 2) + λ * β∞)

    solve(model)
    return getvalue(β)
end


function L1_projection(v; z=1.0)
    if norm(v, 1) ≤ z
        return v
    else
        return sign.(v) .* simplex_projection(abs.(v); z=z)
    end
end


"""
Projection onto the probabilistic simplex in O(n), from
https://web.stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
"""
function simplex_projection(v; z=1.0)
    n = length(v)
    U = collect(1:n)
    s = 0
    ρ = 0

    while length(U) > 0
        k = sample(U, 1)[1]
        G = filter(j -> v[j] ≥ v[k], U)
        L = filter(j -> v[j] < v[k], U)

        Δρ = length(G)
        Δs = sum([v[j] for j in G])

        if (s + Δs) - (ρ + Δρ) * v[k] < z
            s += Δs
            ρ += Δρ
            U = L
        else
            U = setdiff(G, [k])
        end
    end

    θ = (s - z) / ρ
    return max.(v .- θ, 0)
end


# Utils
# ==============================================================================
"Full LP implementation. For benchmarking and testing."
function baseline_group_dantzig(X, y, g, λ; args...)
    # TODO Assumes g has groups 1:k; shouldn't need this assumption
    n, p = size(X)
    k = length(unique(g))
    solver = construct_solver(; args...)
    model = Model(solver=solver)

    @variables model begin
        β⁺[1:p] ≥ 0
        β⁻[1:p] ≥ 0
        βg[1:k]
        ∇⁺[1:p] ≥ 0
        ∇⁻[1:p] ≥ 0
        r[1:n]
    end

    @constraints model begin
        r .== y - X * (β⁺ - β⁻)
        ∇⁺ .- ∇⁻ .== X'r
    end

    for i in 1:k
        group_jx = find(g .== i)
        @constraint(model, βg[i] .≥ β⁺[group_jx] + β⁻[group_jx])
        @constraint(model, sum(∇⁺[group_jx] .+ ∇⁻[group_jx]) ≤ λ)
    end

    obj = @objective(model, Min, sum(βg))

    solve(model)
    β = sparse(getvalue(β⁺) .- getvalue(β⁻))
    return β, model, nothing
end


function group_dantzig_example(n, p, k, density; SNR=10)
    X, _, _ = regression_example(n, p, density)
    groups = rand(1:k, p)

    n_eff_groups = Int(ceil(k * density))
    eff_groups = sortperm(rand(k))[1:n_eff_groups]

    is_eff_group = [g ∈ eff_groups for g in groups]
    true_coeffs = ifelse.(is_eff_group, rand(p) .* 10, 0)
    clean_y = X * true_coeffs
    noisy_y = clean_y + randn(n) * var(clean_y) / SNR
    return X, noisy_y, groups, true_coeffs
end
