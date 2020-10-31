using LinearAlgebra
using StatsBase
using PyCall
using Plots
pyplot(fmt=:png)
include("../src/kernel.jl")
datasets = pyimport("sklearn.datasets")
cluster = pyimport("sklearn.cluster")
neighbors = pyimport("sklearn.neighbors")

# circle
n_samples = 1500
factor = 0.3
noise = 0.05
random_state = 0
X_2d, _ = datasets.make_circles(n_samples; factor=factor, noise=noise, random_state=random_state)
num_class = 2

# flatten
X_1d = [X_2d[1, :]]
for i in 2:size(X_2d)[1]
    push!(X_1d, X_2d[i, :])
end

# kenrel
τ = 1 
σ = 0.1
k = GaussianKernel(τ, σ)

# Lhaplacian Eigen Map
K = kernel_matrix(k, X_1d)
Λ = diagm(vec(sum(K, dims=1)))
P = Λ - K
f = eigen(P, Λ)
eigen_values = f.values
eigen_vectors = f.vectors

X_lem = eigen_vectors[:, 2:3]
y = zeros(Int8, size(X_lem)[1])

for i=1:n_samples
    x = X_lem[i, :]
    y[i] = Int8(x[1] > 0) + 1
end

# plot
c1 = findall(x -> x == 1, y) 
c2 = findall(x -> x == 2, y) 
plot(X_2d[c1, 1], X_2d[c1, 2];
     st=:scatter, label="cluster 1", ms=5, size=(400, 400), alpha=0.3)
plot!(X_2d[c2, 1], X_2d[c2, 2];
     st=:scatter, label="cluster 2", ms=5, size=(400, 400), alpha=0.3)

savepath = @sprintf("%s/spectral_(%s,%s).png", @__DIR__, τ, σ)
savefig(savepath)