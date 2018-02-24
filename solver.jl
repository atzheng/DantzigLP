"""
The main engine for fast Dantzig estimation. Handles column and constraint
generation, solving for λ paths, and diagnostics.

For each type of Dantzig problem, we need specialized methods for the following
functions (dispatched on types inheriting from DantzigModel):

Constraint generation:
- add_Xtr_constraint!
- get_constraint_violations

Column generation:
- add_beta!
- get_reduced_costs
"""


"""
Struct to hold constraint refs and mutable variables for column
and constraint generation. Extend for specialized Dantzig problems.
"""
abstract type DantzigModel end
Base.show(io::IO, x::DantzigModel) = show(io, x.gurobi_model)


function solve_dantzig_lp!(model, λ, initial_soln;
                           column_generation = true, max_columns = 40,
                           constraint_generation = true, max_constraints = 50,
                           initialize_basis = false,
                           verbose = false, timeout = Inf,
                           solver_params = Dict(),
                           tol = 1e-9)

    vinfo(msg) = verbose_info(verbose, msg)

    vinfo("Initializing columns and constraints...")
    start_time = time_ns()
    # Initialize
    # --------------------------------------------------------------------------
    solver = construct_solver(verbose = verbose, tol = tol, timeout = timeout,
                              params = solver_params)
    setsolver(model.gurobi_model, solver)

    if column_generation
        initialize_columns!(model, initial_soln)
    else
        initialize_columns!(model)
    end

    if constraint_generation
        initialize_constraints!(model, minimum(λ), initial_soln)
    else
        initialize_constraints!(model, minimum(λ))
    end

    if initialize_basis && column_generation && constraint_generation
        initialize_basis!(model, initial_soln)
    end

    solver(modelx, λx) = solve_model(modelx, λx,
                                     column_generation, max_columns,
                                     constraint_generation, max_constraints,
                                     verbose, timeout, tol)

    vinfo(@sprintf("Column and constraint initialization finished in %.2f seconds.", (time_ns() - start_time) / 1e9))

    # Solve
    # --------------------------------------------------------------------------
    if isa(λ, Array)  # Solve a path of deltas
        solutions = []
        for d in λ # TODO enforce an ordering of deltas
            for constr in model.linf_pos_constrs
                JuMP.setRHS(constr, d)
            end
            for constr in model.linf_neg_constrs
                JuMP.setRHS(constr, -d)
            end
            solution = solver(model, d)
            push!(solutions, solution)
        end
        diagnostics = vcat([soln[2] for soln in solutions]...)
        coefs = [soln[1] for soln in solutions]
    else
        coefs, diagnostics = solver(model, λ)
    end
    return (coefs, diagnostics)
end


function solve_model(model, delta,
                     column_generation, max_columns,
                     constraint_generation, max_constraints,
                     verbose, timeout, tol)
    # Intialize tracking for diagnostics
    start_time = time_ns()
    columns_generated = 0
    column_generation_seconds = 0
    constraint_generation_seconds = 0
    constraints_generated = 0
    gurobi_seconds = 0
    initial_vars = union(model.pos_beta_indices, model.neg_beta_indices)
    solve_status, solve_time = @timed solve(model.gurobi_model)
    @show solve_status

    if solve_status == :Infeasible
        error("Infeasible initial model.")
    end

    gurobi_seconds += solve_time
    status = :InProgress

    # iterations = []

    # Start solver
    while ((status == :InProgress) &&
           (column_generation || constraint_generation))

        seconds_elapsed = (time_ns() - start_time) / 1.0e9
        if seconds_elapsed > timeout
            status = :Timeout
            break
        end

        # feas = norm(model.X'getvalue(model.residuals), Inf) - delta
        # pct_feas = (norm(model.X'getvalue(model.residuals), Inf) - delta) / delta
        # obj = (norm(getvalue(model.pos_betas), 1) +
        #        norm(getvalue(model.neg_betas), 1))
        # iteration_log = DataFrame(seconds = seconds_elapsed,
        #                           feas = feas,
        #                           pct_feas = pct_feas,
        #                           obj = obj)
        # push!(iterations, iteration_log)

        # Constraint generation
        if constraint_generation
            congen_start_ts = time_ns()
            pos_constrs = []
            neg_constrs = []

            new_pos_constrs, new_neg_constrs =
                generate_constraints!(model, delta, max_constraints, tol)
            push!(pos_constrs, new_pos_constrs)
            push!(neg_constrs, new_neg_constrs)

            while length(new_pos_constrs) + length(new_neg_constrs) > 0
                solve_status, solve_time = @timed solve(model.gurobi_model)
                @show solve_status
                if solve_status == :Infeasible
                    break
                end
                gurobi_seconds += solve_time
                new_pos_constrs, new_neg_constrs =
                    generate_constraints!(model, delta, max_constraints, tol)
                push!(pos_constrs, new_pos_constrs)
                push!(neg_constrs, new_neg_constrs)
            end

            flat_pos_constrs = vcat(pos_constrs...)
            flat_neg_constrs = vcat(neg_constrs...)

            n_new_constrs = length(flat_pos_constrs) + length(flat_neg_constrs)
            constraints_generated += n_new_constrs
            if verbose info(@sprintf("Generated %d constraints.",
                                     n_new_constrs)) end

            constraint_generation_seconds +=
                (time_ns() - congen_start_ts) / 1.0e9
        end

        # Column generation
        if column_generation
            colgen_start_ts = time_ns()
            new_columns = generate_columns!(model, max_columns, tol)
            if verbose info(@sprintf("Generated %d columns.",
                                    length(new_columns))) end
            if length(new_columns) == 0
                status = :Optimal
            else
                solve_status, solve_time = @timed solve(model.gurobi_model)
                @show solve_status
                gurobi_seconds += solve_time
                columns_generated += length(new_columns)
            end
            column_generation_seconds += (time_ns() - colgen_start_ts) / 1.0e9
        else
            status = :Optimal
        end
    end

    # Extract solution
    beta_values = spzeros(model.size[2])
    for (var, i) in zip(model.pos_betas, model.pos_beta_indices)
        beta_values[i] = getvalue(var)
    end

    for (var, i) in zip(model.neg_betas, model.neg_beta_indices)
        beta_values[i] -= getvalue(var)
    end

    # Tally up diagnostics
    end_time = time_ns()
    total_seconds = (end_time - start_time) / 1.0e9
    correct_initial_vars = length(intersect(beta_values.nzind, initial_vars))
    total_vars = length(beta_values.nzind)
    diagnostics = DataFrame(
        delta = delta,
        total_seconds = total_seconds,
        gurobi_seconds = gurobi_seconds,
        columns_generated = columns_generated,
        column_generation_seconds = column_generation_seconds,
        constraints_generated = constraints_generated,
        constraint_generation_seconds = constraint_generation_seconds,
        initial_vars = length(initial_vars),
        correct_initial_vars = correct_initial_vars,
        total_vars = total_vars,
        status = string(status))
        # iterations = vcat(iterations...))

    return beta_values, diagnostics
end

# Column / Constraint Generation
# ==============================================================================
function generate_columns!(model::DantzigModel, max_columns, tol)
    # Compute reduced costs
    pos_costs, neg_costs = get_reduced_costs(model)
    pos_costs[model.pos_beta_indices] = Inf
    neg_costs[model.neg_beta_indices] = Inf

    max_columns = min(max_columns, length(pos_costs))
    pos_min_indices = sortperm(vec(pos_costs))[1:max_columns]
    neg_min_indices = sortperm(vec(neg_costs))[1:max_columns]

    # Sort reduced cost vectors
    # (idx, sgn) tuples for return values
    pos_return_vals = zip_collect(
        pos_min_indices, ones(length(pos_min_indices)))
    neg_return_vals = zip_collect(
        neg_min_indices, -ones(length(neg_min_indices)))

    pos_min_costs = zip_collect(pos_costs[pos_min_indices], pos_return_vals)
    neg_min_costs = zip_collect(neg_costs[neg_min_indices], neg_return_vals)

    all_costs = vcat(pos_min_costs, neg_min_costs)
    new_columns = [x[2] for x in all_costs if x[1] < -tol]

    # Add betas
    for (idx, sign) in new_columns
        add_beta!(model, idx, Integer(sign))
    end

    return new_columns
end


function add_beta!(model::DantzigModel, idx::Union{UnitRange, Vector},
                   sgn::Vector)
    for (i, s) in zip(idx, sgn)
        add_beta!(model, i, s)
    end
end


function generate_constraints!(model::DantzigModel, delta,
                               max_constraints, tol)
    # TODO should track the constraint indices so the tolerance won't matter
    n, p = model.size

    constraint_values = get_constraint_violations(model)
    constraint_indices = sortperm(abs.(constraint_values), rev = true)

    new_pos_constrs = []
    new_neg_constrs = []
    for row in constraint_indices[1:min(max_constraints, p)]
        val = constraint_values[row]
        if val < -delta - tol
            new_constr = add_Xtr_constr!(model, delta, row, -1)
            push!(new_neg_constrs, new_constr)
        elseif val > delta + tol
            new_constr = add_Xtr_constr!(model, delta, row, 1)
            push!(new_pos_constrs, new_constr)
        end
    end
    return new_pos_constrs, new_neg_constrs
end


function add_Xtr_constr!(model::DantzigModel, delta::Number,
                         idx::Union{UnitRange, Vector}, sgn::Vector)
    for (i, s) in zip(idx, sgn)
        add_Xtr_constr!(model, delta, i, s)
    end
end


# Initialization Functions
# ==============================================================================
function initialize_columns!(model)
    n, p = model.size
    init_cols = 1:p
    add_beta!(model, init_cols, fill(1, p))
    add_beta!(model, init_cols, fill(-1, p))
end


function initialize_columns!(model, initial_soln)
    for idx in initial_soln.nzind
        coef = initial_soln[idx]
        # Only need to add one sign, but that doesn't work with an array
        # of deltas. TODO
        new_var = add_beta!(model, idx, Integer(sign(coef)))
        new_var = add_beta!(model, idx, Integer(-sign(coef)))
        setvalue(new_var, coef)
    end
end


function initialize_constraints!(model, λ)
    n, p = model.size
    init_constrs = 1:p
    add_Xtr_constr!(model, λ, init_constrs, fill(1, p))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, p))
end


function initialize_constraints!(model, λ, initial_soln)
    init_constrs = initial_soln.nzind
    n_nz = length(initial_soln.nzind)
    add_Xtr_constr!(model, λ, init_constrs, fill(1, n_nz))
    add_Xtr_constr!(model, λ, init_constrs, fill(-1, n_nz))
end
