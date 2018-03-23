using DantzigLP, Lasso, DataFrames, CSV


# Load Data
# ==============================================================================
dataset = ARGS[1]
DATA_DIR = "/pool001/dantzig/"
function load_data(fname)
    return readdlm(joinpath(DATA_DIR, dataset, fname), ',')
end


function get_normalizer(X)
    colnorms = mapslices(x -> norm(x, 2), X, [1])
    keep_columns = find(colnorms .!= 0)
    function normalize(X2)
        X2[:, keep_columns] ./ colnorms[:, keep_columns]
    end
end


X_raw = load_data("X_training.csv")
y_raw = load_data("y_training.csv")

X_normalizer = get_normalizer(X_raw)
y_normalizer = get_normalizer(y_raw)

X_train = X_normalizer(X_raw)
y_train = y_normalizer(y_raw) |> vec

X_valid = load_data("X_validation.csv") |> X_normalizer
y_valid = load_data("y_validation.csv") |> y_normalizer |> vec

X_test = load_data("X_testing.csv") |> X_normalizer
y_test = load_data("y_testing.csv") |> y_normalizer |> vec


# Fit Models
# ==============================================================================
n, p = size(X_train)

# Warmup runs
lasso = fit(LassoPath, X_train, y_train)
λs = lasso.λ * n
dlp_results = DantzigLP.dantzig_lp(X_train, y_train, λs[10])


# Begin Logging
lasso_t = @elapsed lasso = fit(LassoPath, X_train, y_train)

dlp_t = @elapsed dlp_results = DantzigLP.dantzig_lp(
    X_train, y_train, λs; verbose=true)
dlp_coefs = hcat(dlp_results[1]...)


# Validate and test
# ==============================================================================
function compute_mse(X, y, coefs)
    mean((X * coefs .- y) .^ 2, [1])
end


dlp_mse = compute_mse(X_valid, y_valid, dlp_coefs)
_, dlp_idx = findmin(dlp_mse)
dlp_best = dlp_coefs[:, dlp_idx]

lasso_mse = compute_mse(X_valid, y_valid, lasso.coefs)
_, lasso_idx = findmin(lasso_mse)
lasso_best = lasso.coefs[:, lasso_idx]

dlp_error = compute_mse(X_test, y_test, dlp_best)[1]
lasso_error = compute_mse(X_test, y_test, lasso_best)[1]

results = DataFrame(algo=["DantzigLP", "Lasso"],
                    L0=[norm(dlp_best, 0), norm(lasso_best, 0)],
                    mse=[dlp_error, lasso_error],
                    time=[dlp_t, lasso_t],
                    dataset=dataset,
                    n=n, p=p)

CSV.write(@sprintf("real-world/%s.csv", dataset), results)
