using JuMP, Gurobi, MathProgBase, Lasso, DataFrames

TOL = 1e-8

type BasicDantzigModel <: DantzigModel
    gurobi_model     # Gurobi model object (mutable)
    size             # (n, p) tuple
    pos_betas        # Postive Beta variables already in the model (mutable)
    pos_beta_indices # Indices corresponding to entries of pos_betas (mutable)
    neg_betas        # Negative Beta variables already in the model (mutable)
    neg_beta_indices # Indices corresponding to entries of neg_betas (mutable)
    residuals        # Residual variables
    residual_constrs # Constraints r = y - X * Beta
    linf_pos_constrs # Constraints X'r <= delta
    linf_neg_constrs # Constraints X'r >= -delta
    X                # Data matrix X
end

function BasicDantzigModel(X, y)
    n, p = size(X)
    gurobi_model = Model()

    residuals = @variable(gurobi_model, [1:n])
    residual_constrs = @constraint(gurobi_model, y - residuals .== 0)
    obj = @objective(gurobi_model, Min, 0)

    model = BasicDantzigModel(gurobi_model, (n, p), [], [], [], [],
                              residuals, residual_constrs, [], [], X)
    return model
end


"""
Baseline solution. Mainly for testing.
Under normal conditions, should be identical to dantzig_lp with
column / constraint generation disabled.
"""
function baseline_dantzig(X, y, λ, verbose = false)
    n, p = size(X)

    model = Model(solver = GurobiSolver(OutputFlag = ifelse(verbose, 1, 0)))

    residuals = @variable(model, [1:n])
    abs_beta_pos = @variable(model, [1:p], lowerbound = 0)
    abs_beta_neg = @variable(model, [1:p], lowerbound = 0)

    residual_constrs =
        @constraint(model, y - X * (abs_beta_pos - abs_beta_neg) .== residuals)
    linf_pos_constrs = @constraint(model, X' * residuals .<= λ)
    linf_neg_constrs = @constraint(model, X' * residuals .>= -λ)

    obj = @objective(model, Min, sum(abs_beta_pos + abs_beta_neg))

    if verbose info("Solving model...") end
    solve(model)
    return model, sparse(getvalue(abs_beta_pos) - getvalue(abs_beta_neg))
end


function dantzig_lp(X, y, λ; initializer_fn = lasso_initializer, args...)
    model = BasicDantzigModel(X, y)
    initial_soln, initializer_seconds = @timed initializer_fn(X, y, λ)
    soln, log = solve_dantzig_lp!(model, λ, initial_soln; args...)
    return soln, model, log
end


function initialize_basis!(model::BasicDantzigModel, initial_soln)
    n, p = model.size
    current_n = n + 2 * length(initial_soln.nzind)

    JuMP.build(model.gurobi_model)

    grb_model = internalmodel(model.gurobi_model).inner
    vbasis = zeros(current_n)
    cbasis = - ones(current_n)
    Gurobi.set_intattrarray!(grb_model, "VBasis", 1, current_n, vbasis)
    Gurobi.set_intattrarray!(grb_model, "CBasis", 1, current_n, cbasis)

    Gurobi.update_model!(grb_model)
end


# Column / Constraint Generation
# ==============================================================================
"""
Closure to add new beta variables to the model.
WARNING: mutates betas and beta_indices.
"""
function add_beta!(model::BasicDantzigModel, idx::Integer, sgn::Number)
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
            objective = 1,
            inconstraints = model.residual_constrs,
            coefficients = -sgn * full(model.X[:, idx]),
            lowerbound = 0)
        push!(beta_indices, idx)
        push!(betas, new_var)
        return new_var
    end
end

function get_reduced_costs(model::BasicDantzigModel)
    n, p = model.size

    # Compute reduced costs separately for the positive and negative
    # components of Beta
    # First n constraints correspond to the β constraints.
    # TODO Shouldn't need to make this assumption....
    duals = model.gurobi_model.linconstrDuals[1:n]
    pos_reduced_costs = 1 + duals'model.X
    neg_reduced_costs = 1 - duals'model.X
    return (pos_reduced_costs, neg_reduced_costs)
end


function get_constraint_violations(model::BasicDantzigModel)
    resid_vals = getvalue(model.residuals)
    return vec(resid_vals'X)
end


function add_Xtr_constr!(model::BasicDantzigModel, delta::Number,
                         idx::Integer, sign::Integer)
    if sign == -1
        new_constr = @constraint(
            model.gurobi_model,
            dot(model.X[:, idx], model.residuals) >= -delta)
        push!(model.linf_neg_constrs, new_constr)
    elseif sign == 1
        new_constr = @constraint(
            model.gurobi_model,
            dot(model.X[:, idx], model.residuals) <= delta)
        push!(model.linf_pos_constrs, new_constr)
    end
    return new_constr
end


# Initializers
# ==============================================================================
function lasso_initializer(X, y, delta::Number; args...)
    return lasso_initializer(X, y, [delta], args...)
end


function lasso_initializer(X, y, delta::Array; args...)
    n, p = size(X)
    soln = fit(LassoPath, X, y, λ = delta ./ n;
               maxncoef = p, standardize = false, cd_tol = 1e-15, args...,
               ).coefs
    return merge_vectors([soln[:, j] for j in 1:size(soln, 2)]...)
end


function max_correlation_initializer(X, y, delta;
                                     coefs = Int(round(0.2 * length(y))))
    n, p = size(X)
    Xty = X'y
    nzind = sortperm(abs.(Xty), rev = true)[1:coefs]
    results = spzeros(p)
    results[nzind] = Xty[nzind]
    return results
end


function composite_initializer(X, y, delta, initializer_fns)
    initial_indices = [initializer_fn(X, y, delta)
                       for initializer_fn in initializer_fns]
    return merge_vectors(initial_indices...)
end


function recursive_initializer(X, y, delta, initializer_fn, max_depth)
    soln = initializer_fn(X, y, delta)
    if max_depth == 1
        return soln
    else
        p = size(X, 2)
        next_indices = sort(setdiff(1:p, soln.nzind))
        next_vector = spzeros(p)
        next_results = recursive_initializer(
            X[:, next_indices], y, delta, initializer_fn, max_depth - 1)
        next_vector[next_indices] = next_results
        return merge_vectors(soln, next_vector)
    end
end
