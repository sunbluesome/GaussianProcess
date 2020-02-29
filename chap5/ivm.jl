include("../src/kernel.jl")
include("../src/gp.jl")

using Printf
using Plots
using Random
pyplot(fmt=:png)
using Plots.PlotMeasures
using LaTeXStrings

# data preparation
Random.seed!(20200208)
xtrain = vcat(rand(80),rand(20) * 3 .+ 1.0)
sort!(xtrain)
ytrue = sin.(xtrain*2)
ytrain = ytrue + randn(length(xtrain)) * 0.3
xtest = collect(-1:0.01:5)

# GP parameters
τ = 1.
σ = 1.
η = 1.

# Inducing Variables
Ms = [2, 5, 10]
ivs = []
for i in 1:3
    iv_base = 5 / Ms[i]
    iv = [iv_base * j for j in 0:Ms[i]-1]
    push!(ivs, iv)
end

k = GaussianKernel(τ, σ)
gp = GaussianProcess(k, η)

# plot GP with whole data
μs_whole, σs_whole = predict(gp, xtest, xtrain, ytrain)
p = plot(xtest, sin.(2xtest); label=L"$\sin(2x)$", color=:black)
plot!(xtest, μs_whole-2sqrt.(σs_whole); label="", alpha=0, fill=μs_whole+2sqrt.(σs_whole), fillalpha=0.3, color=:blue)
plot!(xtest, μs_whole; st=:line, label="gp (μ±2σ)", color=:blue, ylim=(-3,4))

# plot SoD
P = []
P = push!(P, p)
for i in 1:3
    method = InducingVariableMethod(ivs[i])
    μs,σs = predict(gp, xtest, xtrain, ytrain, method)

    # gp (whole data)
    p = plot(xtest, μs_whole-2sqrt.(σs_whole); label="", alpha=0, fill=μs_whole+2sqrt.(σs_whole), fillalpha=0.3, color=:blue)
    plot!(xtest, μs_whole; st=:line, label="gp (μ±2σ)", color=:blue)
    plot!(xtrain, ytrain; st=:scatter, label="", ms=5, color=:blue)

    # gp (Inducing variable method)
    plot!(xtest, μs-2sqrt.(σs); label="", alpha=0, fill=μs+2sqrt.(σs), fillalpha=0.3, color=:red)
    plot!(xtest, μs; st=:line, label="IVM (μ±2σ)", color=:red)
    plot!(ivs[i], -2 .* ones(length(ivs[i])); st=:scatter, label="", ms=5, marker=:x, color=:black, ylim=(-3,4))
    push!(P,p)
end
plot(P..., layout=(2,2), size=(800,600))

savepath = @sprintf("%s/ivm.png", @__DIR__)
savefig(savepath)