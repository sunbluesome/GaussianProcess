using LinearAlgebra
using StatsBase
using PyCall
using Plots
using ProgressMeter
pyplot(fmt=:png)
include("../src/kernel.jl")
datasets = pyimport("sklearn.datasets")

# circle
n_samples = 1500
factor = 0.3
noise = 0.05
random_state = 0
X_2d, _ = datasets.make_circles(n_samples; factor=factor, noise=noise, random_state=random_state)
num_class = 2
rng = MersenneTwister(0)
y = rand(rng, collect(1:num_class), n_samples)

# flatten
X_1d = [X_2d[1, :]]
for i in 2:size(X_2d)[1]
    push!(X_1d, X_2d[i, :])
end

# kenrel
# τ = 1.0
# σ = 0.01
for τ in [1, 3, 10, 100]
    for σ in [1.0, 0.3, 0.1, 0.03, 0.01]
        rng = MersenneTwister(0)
        y[:] .= rand(rng, collect(1:num_class), n_samples)

        k = GaussianKernel(τ, σ)
        K = kernel_matrix(k, X_1d)

        dist = repeat(diag(K), outer=(1, num_class))
        @showprogress 1 "epoch : " for epoch in 1:100
            for i in 1:num_class
                dist[:, i] -= 2 * mean(K[:, y .== i], dims=2)
                dist[:, i] .+= 2 * mean(K[y .== i, :][:, y .== i])
                y[:] .= vec([ind[2] for ind in argmin(dist, dims=2)])
            end
        end
        print(counts(y))

        # plot
        inds = findall(x -> x == 1, y)
        plot(X_2d[inds, 1], X_2d[inds, 2];
            st=:scatter, label="cluster 1", ms=5, size=(400, 400), alpha=0.3)
        inds = findall(x -> x == 2, y)
        plot!(X_2d[inds, 1], X_2d[inds, 2];
            st=:scatter, label="cluster 2", ms=5, size=(400, 400), alpha=0.3)

        savepath = @sprintf("%s/kkmeans_(%s, %s).png", @__DIR__, τ, σ)
        savefig(savepath)
    end
end