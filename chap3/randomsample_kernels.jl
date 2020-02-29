include("../src/kernel.jl")
include("../src/gp.jl")

using Plots
pyplot()
using Plots.PlotMeasures
using LaTeXStrings

kernels = [
    LinearKernel(randn()),
    GaussianKernel(1,1),
    ExponentialKernel(0.5),
    PeriodicKernel(1,0.5),
    Matern3Kernel(2),
    Matern5Kernel(2)
]

titles = [
    "linear",
    "rbf", 
    "exponential", 
    "periodic", 
    "matern3", 
    "matern5"
]


num_sample=200
num_resample=5
P = []
for (i,k) in enumerate(kernels)
    x = collect(range(-4,stop=4,length=num_sample))
    K = kernel_matrix(k,x)
    
    y = sample(k, x)

    xlim = [minimum(x)-0.1, maximum(x)+0.1]
    ylim = [minimum(y) , maximum(y)]
    p1 = plot(x, y, label=@sprintf("y%d", 1))
    
    for j=2:num_resample
        y = sample(k, x)
        if minimum(y) < ylim[1]
            ylim[1] = minimum(y)
        end
        if maximum(y) > ylim[2]
            ylim[2] = maximum(y)
        end
        plot!(p1, x, y, label=@sprintf("y%d", j))
    end
    ylim[1] -= 0.1
    ylim[2] += 0.1
    
    diff_x = xlim[2]-xlim[1]
    diff_y = ylim[2]-ylim[1]
    aspect_ratio = diff_x/diff_y
        
    p1 = scatter!(x, zero(x).+minimum(y).-0.05, marker=:x, aspect_ratio=aspect_ratio,
        xlim=xlim, ylim=ylim, legend=:best, label=L"$x$", title=titles[i])
    p2 = heatmap(K, aspect_ratio=1, yflip=true, grid=false, border=nothing,
        xlim=(0.5, size(K)[2]+0.5), ylim=(0.5, size(K)[1]+0.5), title="Kernel Matrix")
    push!(P, p1, p2)
end
plot(P..., size=(600,300*length(kernels)), margin=3mm, layout=(length(kernels),2))
savefig("randomsample2d_kernels.png")