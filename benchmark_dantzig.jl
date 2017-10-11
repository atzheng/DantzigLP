using DataFrames, Gadfly

function run_instance(n, p, delta, fn)
    X, y, actual = generate_regression_example(n, p, 0.2)
    _, time, _ = @timed fn(X, y, delta)
    return time
end

function congen_dantzig(X, y, delta)
    return DantzigLP.colgen_dantzig(
        y, X, delta, column_generation = false, constraint_generation = true)
end

ns = [100, 200, 500, 1000]
ps = [100, 500, 1000, 5000, 10000]
results = Dict((n, p) => run_instance(n, p, 10, congen_dantzig)
               for n in ns for p in ps)

n, p = zip(keys(results)...) |> collect
df = DataFrame(n = collect(n), p = collect(p), time = collect(values(results)))
plot(df, x = :p, y = :time, color = :n, group = :n, Geom.point, Geom.line)
