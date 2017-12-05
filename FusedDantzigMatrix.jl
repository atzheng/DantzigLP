import Base.getindex, Base.size, Base.sort


type FusedDantzigMatrix <: AbstractMatrix{Number}
    n::Integer
    k::Integer
    # Stored values for fast computation
    cum_sums :: Vector  # Cumulative sums up to order k (n x k Array)
    U :: Array{Number, 2}  # From SVD of first k columns of H (n x k Array)
end

size(A :: FusedDantzigMatrix) = (A.n, A.n - A.k)


"""Typical constructor"""
function FusedDantzigMatrix(n::Int, k::Int)
    X1 = ones(n)
    cum_sums = cumsum_k(X1, k - 1)

    Xa = hcat([shift(cumsum_k(ones(n), j), j) for j in 0:(k - 1)]...)
    U = svd(Xa)[1]
    return FusedDantzigMatrix(n, k, cum_sums, U)
end


function getindex(A::FusedDantzigMatrix, i::Colon, j::Integer)
    return vec(getindex(A, :, [j]))
end


"""O(n) multiplications of the form y = Ax for k = 0"""
function Ax(A::FusedDantzigMatrix, x::Vector)
    n, p = size(A)
    results = zeros(n)
    row_1 = -(1 .- collect(1:(n - 1)) ./ n)
    results[1] = dot(row_1, x)
    for i in 2:n
        results[i] = results[i - 1] + (A[i, i - 1] - A[i - 1, i - 1]) * x[i - 1]
    end
    return results
end


"""O(n) multiplications of the form y = x'A for k = 0"""
function xtA(x::Vector, A::FusedDantzigMatrix)
    n, p = size(A)

    left_sum = 0
    right_sum = sum(x)

    left_sums = zeros(p)
    right_sums = zeros(p)

    for i in 1:p
        left_sum += x[i]
        right_sum -= x[i]

        left_sums[i] = left_sum * (1 - i / n)
        right_sums[i] = right_sum * i / n
    end

    diffs = right_sums .- left_sums
    return RowVector(diffs)
end


function getindex(A::FusedDantzigMatrix, i::Colon, j::Vector)
    if A.k == 1
        upper = [- ones(jx) * (1 - jx / A.n) for jx in j]
        lower = [ones(A.n - jx) * (jx / A.n) for jx in j]
        Aj = hcat([vcat(u, l) for (u, l) in zip(upper, lower)]...)
    else
        if maximum(j) > A.n - A.k
            throw(BoundsError())
        else
            sorted_j = sort(j) .- 1 .+ A.k
            diffs = sorted_j - shift(sorted_j, 1)
            Xj = hcat(accumulate((vec, i) -> shift(vec, i),
                                 A.cum_sums, diffs)...)
            Aj = invdiff_matvecmult(A, Xj)
        end
    end
    return Aj
end


"""kth order cumulative sum"""
function cumsum_k(x::AbstractVector, k::Integer)
    if k == 0
        return x
    else
        return cumsum(cumsum_k(x, k - 1))
    end
end


function getindex(A :: FusedDantzigMatrix, i :: Integer, j :: Integer)
    if A.k == 1
        if i > A.n | j > (A.n - 1)
            throw(BoundsError())
        elseif i <= j
            return - (1 - j / A.n)
        else
            return j / A.n
        end
    else
        Xj = shift(A.cum_sums, A.k + j - 1)
        return (Xj - A.U * (A.U'Xj))[i]
    end
end


# TODO this is poorly named
function invdiff_matvecmult(A::FusedDantzigMatrix, x::Array)
    return x - A.U * (A.U'x)
end
