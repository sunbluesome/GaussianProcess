using Plots
pyplot(fmt=:png)
using Plots.PlotMeasures
using LaTeXStrings
using Optim
include("../src/kernel.jl")
include("../src/gp.jl")
import DelimitedFiles: readdlm

# objective functions
function objfun(gp, xtrain, ytrain , params)
    update!(gp, [gp.k.θ1, params...])
    loglikelihood(gp, xtrain, ytrain)
end

function objfunc_deriv(storage, gp, xtrain, ytrain, params)
    update!(gp, [gp.k.θ1, params...])
    storage[1] = loglikelihood_deriv(gp, xtrain, ytrain)[2]
    storage[2] = loglikelihood_deriv(gp, xtrain, ytrain)[3]
end

# data 
path = @sprintf("%s/gpr.dat", @__DIR__)
train = readdlm(path, '\t', Float64, '\n')
xtrain = train[:,1]
ytrain = train[:,2]
xtest = collect(range(-1,stop=4,length=100))

# GP parameters
τ = 1.
σ = 1.
η = 1.

k = GaussianKernel(τ, σ)
gp = GaussianProcess(k, η)
method = SubsetOfData()
x0 = [σ, η] 

# Optimize
f(x) = -objfun(gp, xtrain , ytrain, x)
g!(storage, x) = -objfunc_deriv(storage, gp, xtrain, ytrain, x)

lower = [0., 0.]
upper = [10., 10.]
res = optimize(f, g!, lower, upper, x0, Fminbox(NelderMead()))

# plot
x0 = [1.0, σ, η] 
xopt = [1., Optim.minimizer(res)...]
P = []
for x in [x0, xopt]
    τ, σ, η = x 
    k = GaussianKernel(τ, σ)
    gp = GaussianProcess(k, η)
    μs,σs = predict(gp, xtest, xtrain, ytrain, method)
    
    p = plot(xtest, μs-2sqrt.(σs), label="±2σ", alpha=0,
        fill=μs+2sqrt.(σs), fillalpha=0.3, color=:red)
    plot!(xtrain, ytrain, st=:scatter, label="train", ms=5, color=:blue)
    plot!(xtest, μs, st=:line, label="gp", lw=3, ls=:dash, color=:red)
    push!(P, p)
end
plot(P..., size=(1200,400), margin=3mm, layout=(1,2), ylim=(-3,4))
savepath = @sprintf("%s/estimate.png", @__DIR__)
savefig(savepath)