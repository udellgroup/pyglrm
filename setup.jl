# setup.jl

function ensure(package::String)  # Install package if not installed.
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

# This sets the version of the packages that are used.  In the long term it
# would be better to use built in Julia commands.  However, here we pin the
# versions to specific git commit numbers and thus perform this manually.
shell_file = ARGS[2]
run(`bash $shell_file`)

# Finally, we build the packages now since PyCall needs to be built with
# ENV["PYTHON"] set and LowRankModels takes so long that it may cause users to
# think the system is not working upon first use.
Pkg.build("LowRankModels")
Pkg.build("PyCall")
