include("../src/kernel.jl")
include("../src/gp.jl")

using Plots
pyplot()
using Plots.PlotMeasures
using LaTeXStrings

θ1=1
θ2=1
k = GaussianKernel(θ1, θ2)

num_samples = [4,7,16,50]
P = []
for n in num_samples
    x = range(1,stop=4,length=n)
    y = range(1,stop=4,length=n)
    xx = [[i,j] for i in x for j in y]
    
    K = kernel_matrix(k, xx)   # for plotting
    z = reshape(sample(k, xx), length(y), length(x))
    
    p1 = plot(x, y, z, st=:surface, label=L"$y$", xlabel=:x, ylabel=:y, zlabel=:z)
    push!(P, p1)
end
plot(P..., size=(600,300*length(num_samples)), margin=3mm, layout=(length(num_samples),1))
savefig("randomsample2d.png")