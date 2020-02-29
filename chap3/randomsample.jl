include("../src/kernel.jl")
include("../src/gp.jl")

using Plots
pyplot()
using Plots.PlotMeasures
using LaTeXStrings

θ1=1
θ2=1
k = GaussianKernel(θ1, θ2)

num_samples = [4,7,16,100]
P = []
for n in num_samples
    x = collect(range(1,stop=4,length=n))
    K = kernel_matrix(k, x)   # for plotting
    y = sample(k, x)
    
    
    xlim = (minimum(x)-0.1, maximum(x)+0.1)
    ylim = (minimum(y)-0.1, maximum(y)+0.1)
    diff_x = xlim[2]-xlim[1]
    diff_y = ylim[2]-ylim[1]
    aspect_ratio = diff_x/diff_y
    plot(x, y, st=:scatter, label=L"$y$")
    p1 = plot!(x, zero(x).+minimum(y).-0.05, marker=:x, st=:scatter, aspect_ratio=aspect_ratio,
        xlim=xlim, ylim=ylim, legend=:best, label=L"$x$")
    p2 = heatmap(K, aspect_ratio=1, yflip=true, grid=false, border=nothing,
        xlim=(0.5, size(K)[2]+0.5), ylim=(0.5, size(K)[1]+0.5))
    push!(P, p1, p2)
end

plot(P..., size=(600,300*length(num_samples)), margin=3mm, layout=(length(num_samples),2))
savefig("randomsample.png")