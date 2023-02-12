import julia
import subprocess
julia.install()

subprocess.run("julia ./utils/preprocessing.jl", shell=True)