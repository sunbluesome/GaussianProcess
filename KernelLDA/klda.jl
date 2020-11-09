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
n_samples = 100
factor = 0.3
noise = 0.05
random_state = 0
X_2d, y = datasets.make_circles(n_samples; factor=factor, noise=noise, random_state=random_state)
y .+= 1
num_class = 2

# flatten
X_1d = [X_2d[1, :]]
for i in 2:size(X_2d)[1]
    push!(X_1d, X_2d[i, :])
end

# kenrel
τ = 1.0
σ = 0.25
k = GaussianKernel(τ, σ)

# Kernel matrix
K = kernel_matrix(k, X_1d)
# kmatrix_partial(x::AbstractVector{T}) where T<:Real = kernel_matrix(k, X_1d, [x])

ζ = 1e-8
Vw = zero(K)
VB = zero(K)
for l in 1:num_class
    inds = findall(x -> x == l, y)
    nₗ = length(inds)
    k̄ij = mean(kernel_matrix(k, X_1d, X_1d[inds]), dims=2)
    K_partial_w = kernel_matrix(k, X_1d, X_1d[inds]) .- k̄ij
    K_partial_v = k̄ij .- mean(K, dims=2)
    Sₗ = K_partial_w * transpose(K_partial_w) / nₗ

    Vw += (nₗ / n_samples) * Sₗ
    VB += (nₗ / n_samples) * (K_partial_v * transpose(K_partial_v))
end

f = eigen(VB, (Vw + ζ * K))
eigen_values = reverse(f.values)
eigen_vectors = reverse(f.vectors, dims=1)

# mapping into RKHS
X_map = transpose(eigen_vectors[:, 1]) * K

# plot
c1 = findall(x -> x == 1, y) 
c2 = findall(x -> x == 2, y) 
plot(X_map[c1], randn(n_samples) * 0.1;
     st=:scatter, label="cluster 1", ms=5, size=(400, 400), alpha=0.3)
plot!(X_map[c2], randn(n_samples) * 0.1;
     st=:scatter, label="cluster 2", ms=5, size=(400, 400), alpha=0.3)

savepath = @sprintf("%s/mapped_(%s,%s).png", @__DIR__, τ, σ)
savefig(savepath)