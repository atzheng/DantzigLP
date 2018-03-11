function FISTA(problem, L; tol=1e-6, maxiter=1e6,
               line_search=false, line_search_factor=0.5,
               verbose=false)

    n, p = size(problem.X)
    α = 0 # Acceleration parameter
    prox_β = spzeros(p)
    β = spzeros(p)

    obj = fista_objective(problem, β)
    vinfo(msg) = verbose_info(verbose, msg)

    is_converged = false

    if verbose
        progressbar = Progress(Int(maxiter))
    end

    for i in 1:maxiter
        prox_β_new = prox(problem, β, L)

        # Acceleration step
        α_new = (1 + sqrt(1 + 4 * α ^ 2)) / 2
        γ = (1 - α) / α_new
        β_new = (1 - γ) * prox_β_new + γ * prox_β

        obj_new = fista_objective(problem, β_new)
        if norm(β_new - β, 2) ≤ tol && i > 20
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
