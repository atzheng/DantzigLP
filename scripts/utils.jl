function parse_config(str)
    args = split(str, "_")
    split_args = map(x -> split(x, "="), args)
    return Dict([(Symbol(k), parse(v)) for (k, v) in split_args])
end
