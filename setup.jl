# setup.jl

function ensure(package::String)
  if typeof(Pkg.installed(package)) == Void
    Pkg.add(package)
  end
end

# Ensure that Julia is configured with the necessary packages.
ENV["PYTHON"] = ARGS[1] # Setup using "current" version of Python.
ensure("LowRankModels")
ensure("NullableArrays")
ensure("FactCheck")
ensure("PyCall")

shell_file = ARGS[2]
run(`bash $shell_file`)

Pkg.build("LowRankModels")
Pkg.build("PyCall")
