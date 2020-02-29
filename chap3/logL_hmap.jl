using Printf
using Plots
pyplot(fmt=:png)
using Plots.PlotMeasures
using LaTeXStrings
include("../src/kernel.jl")
include("../src/gp.jl")
import DelimitedFiles: readdlm

# data
path = @sprintf("%s/gpr.dat", @__DIR__)
train = readdlm(path, '\t', Float64, '\n')
xtrain = train[:,1]
ytrain = train[:,2]

function loglikelihood(gp, xtrain, ytrain , θ1, θ2, θ3)
    update!(gp, [θ1, θ2, θ3])
    loglikelihood(gp, xtrain, ytrain)
end

# Gaussian process
k = GaussianKernel()
gp = GaussianProcess(k)

# parameters
θ2 = exp.(collect(range(-3, stop=12, length=100)))
θ3 = exp.(collect(range(-10, stop=3, length=100)))
f(θ2, θ3) = loglikelihood(gp, xtrain, ytrain, 1., θ2, θ3)

# log likelihood
xx = [[i,j] for i in θ2 for j in θ3]
z = zeros(length(xx))
for (i,(x,y)) in enumerate(xx)
    z[i] = f(x,y)
end

z = reshape(z, length(θ3), length(θ2))

# plot
replace!(x->x<=-50 ? -50 : x, z)
plot(log.(θ2), log.(θ3), z, st=:surface, 
    label=L"$y$", xlabel=:x, ylabel=:y, zlabel=:z, camera=(-40,60))

savepath = @sprintf("%s/logL_hmap.png", @__DIR__)
savefig(savepath)