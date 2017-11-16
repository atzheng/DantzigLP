module LSBoost


function large_ls_boost(X, y, eps;
                        max_iter = 10000, min_corr = -Inf, verbose = false,
                        init_active_set = size(X)[2],
                        initial_coefs = nothing,
                        full_update_freq = 1,
                        full_update_min_iters = 1)
    n, p = size(X)
    residuals = y
    solutions = []
    Xtr = X'residuals
    XtX = nothing
    active_X = nothing

    # Get initial active set
    if initial_coefs == nothing
        active_idx = sortperm(abs.(Xtr), rev=true)[1:init_active_set]
        coefs = spzeros(p)
    else
        active_idx = find(initial_coefs .!= 0)
        coefs = initial_coefs
    end

    j = 1
    total_full_update_iters = 0
    for i in 0:(max_iter - 1)
        # Stopping conditions
        max_corr = norm(X' * residuals, Inf)
        if max_corr < min_corr[j]
            push!(solutions, copy(coefs))
            j += 1
            if j > length(min_corr)
                break
            end
        end

        # Full updates
        if i % full_update_freq == 0
            full_update_iters = 0
            while true
                full_update_iters += 1
                index, update = get_coefficient_update(X, residuals)
                coefs[index] += eps * update
                residuals -= eps * X[:, index] * update
                if index in active_idx && full_update_iters >= full_update_min_iters
                    break
                elseif !(index in active_idx)
                    if verbose info("Adding index $index") end
                    push!(active_idx, index)
                end
            end
            active_X = X[:, active_idx]
            XtX = active_X'active_X
            Xtr = active_X'residuals
            info("Full update terminated after $full_update_iters iterations")
            info("Total full update iterations: $total_full_update_iters")
            total_full_update_iters += full_update_iters
        end

        # Fast updates. Note that everything here is indexed according to the active set.
        update_idx = indmax(abs.(Xtr))
        update_coef = univariate_ls_fit(active_X[:, update_idx], residuals)
        coefs[active_idx[update_idx]] += eps * update_coef
        residuals -= eps * update_coef * active_X[:, update_idx]
        Xtr -= eps * update_coef * XtX[:, update_idx]

        if verbose
            l1 = round(norm(coefs, 1), 4)
            l0 = norm(coefs, 0)
            mse = round(sum(residuals .^ 2), 4)
            info("Iter $i:\t Var=$update_idx;\t Max corr=$max_corr;\t L1=$l1;\t L0=$l0;\t MSE=$mse")
        end

    end


    info("Total full update iterations: $total_full_update_iters")
    if isa(min_corr, Array)
        return solutions
    else
        return coefs
    end
end


function ls_boost(X, y;
                  eps = nothing, max_iter = 10000, min_corr = -Inf, verbose = false)
    n, p = size(X)
    residuals = y
    coefs = spzeros(p)

    j = 1
    solutions = []

    for i in 1:max_iter
        index, update = get_coefficient_update(X, residuals)

        Xtr = X'residuals
        residuals_update = X[:, index] * update
        Xtr_update = X'residuals_update

        if eps == nothing  # Compute optimal step size
            TOL = 1e-5
            all_eps = (Xtr .- Xtr[index]) ./ (Xtr_update - Xtr_update[index])
            update_eps = minimum(all_eps[!isnan(all_eps)]) + TOL
            info("New epsilon: $update_eps")
        else
            update_eps = eps
        end

        coefs[index] += update_eps * update
        residuals -= update_eps * residuals_update
        max_corr = norm(Xtr, Inf)
        if max_corr < min_corr[j]
            push!(solutions, copy(coefs))
            j += 1
            if j > length(min_corr)
                break
            end
        end

        if verbose
            l1 = round(norm(coefs, 1), 4)
            l0 = norm(coefs, 0)
            mse = round(sum(residuals .^ 2), 4)
            info("Iter $i:\t Var: $index\t Max corr=$max_corr;\t L1=$l1;\t L0=$l0;\t MSE=$mse")
        end
    end

    if isa(min_corr, Array)
        return solutions
    else
        return coefs
    end
end


function get_coefficient_update(X, resids)
    n, p = size(X)
    coefs = mapslices(x -> univariate_ls_fit(x, resids), X, 1)
    preds = broadcast(*, coefs, X)
    resids_new = broadcast(-, resids, preds)
    sq_resids = mapslices(x -> sum(x .^ 2), resids_new, 1)
    best_index = indmin(sq_resids)
    return best_index, coefs[best_index]
end


function univariate_ls_fit(x, y)
    return mean(x .* y) / mean(x .^ 2)
end

end
