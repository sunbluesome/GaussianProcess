include("../src/kernel.jl")
include("../src/gp.jl")

using Printf
using Plots
using Random
pyplot(fmt=:png)
using Plots.PlotMeasures
using LaTeXStrings

# data preparation
Random.seed!(20200207)
xtrain = vcat(rand(80),rand(20) * 3 .+ 1.0)
sort!(xtrain)
ytrue = sin.(xtrain*2)
ytrain = ytrue + randn(length(xtrain)) * 0.3
xtest = collect(-1:0.01:5)

# GP parameters
τ = 1.
σ = 2.
η = 0.1

k = GaussianKernel(τ, σ)
gp = GaussianProcess(k, η)

# subset of data
Ns = [2,5,10]
indices = []
for N in Ns
    q = 100 / (N + 1)
    push!(indices,Int.(floor.([q*i for i in 1:N])))
end

# plot GP with whole data
μs_whole, σs_whole = predict(gp, xtest, xtrain, ytrain)
p = plot(xtest, sin.(2xtest); label=L"$\sin(2x)$", color=:black)
plot!(xtest, μs_whole-2sqrt.(σs_whole); label="", alpha=0, fill=μs_whole+2sqrt.(σs_whole), fillalpha=0.3, color=:blue)
plot!(xtest, μs_whole; st=:line, label="gp (μ±2σ)", color=:blue, ylim=(-3,4))

# plot SoD
P = []
P = push!(P, p)
for i in 1:3
    xtrain_sod = xtrain[indices[i]]
    ytrain_sod = ytrain[indices[i]]

    method = SubsetOfData(indices[i])
    μs,σs = predict(gp, xtest, xtrain, ytrain, method)

    p = plot(xtest, μs_whole-2sqrt.(σs_whole); label="", alpha=0, fill=μs_whole+2sqrt.(σs_whole), fillalpha=0.3, color=:blue)
    plot!(xtest, μs_whole; st=:line, label="gp (μ±2σ)", color=:blue)
    plot!(xtest, μs-2sqrt.(σs); label="", alpha=0, fill=μs+2sqrt.(σs), fillalpha=0.3, color=:red)
    plot!(xtest, μs; st=:line, label="SoD (μ±2σ)", color=:red)
    plot!(xtrain, ytrain; st=:scatter, label="", ms=5, color=:blue)
    plot!(xtrain_sod, -2 .* ones(length(xtrain_sod)); st=:scatter, label="", ms=5, marker=:x, color=:black, ylim=(-3,4))
    push!(P,p)
end
plot(P..., layout=(2,2), size=(800,600))

savepath = @sprintf("%s/sod.png", @__DIR__)
savefig(savepath)