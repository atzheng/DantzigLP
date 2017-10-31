module LSBoost


function ls_boost(X, y, eps; max_iter = 10000, min_corr = Inf, verbose = false)
    n, p = size(X)
    residuals = y
    coefs = spzeros(p)

    j = 1
    solutions = []
    for i in 1:max_iter
        index, update = get_coefficient_update(X, residuals)
        coefs[index] += eps * update
        residuals -= eps * X[:, index] * update
        max_corr = norm(X' * residuals, Inf)

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
            info("Iter $i:\t Max corr=$max_corr;\t L1=$l1;\t L0=$l0;\t MSE=$mse")
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

