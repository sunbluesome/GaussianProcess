using LinearAlgebra
using SparseArrays
using Arpack
using StatsBase
using PyCall
using Plots
using ProgressMeter
pyplot(fmt=:png)
include("../src/kernel.jl")
datasets = pyimport("sklearn.datasets")
cluster = pyimport("sklearn.cluster")
neighbors = pyimport("sklearn.neighbors")

# swiss roll
n_samples = 1500
noise = 0.0
random_state = 0
X_2d, _ = datasets.make_swiss_roll(n_samples; noise=noise, random_state=random_state)

# split into five clusters
connectivity = neighbors.kneighbors_graph(X_2d, n_neighbors=10, include_self=false)
ward = cluster.AgglomerativeClustering(n_clusters=5, connectivity=connectivity, linkage="ward").fit(X_2d)
labels = ward.labels_
inds_clusters = Dict{Int8, Array{Int64, 1}}()
for i in 1:5
    inds_clusters[i] = findall(x -> x == i, labels) 
end

X_1d = [X_2d[1, :]]
for i in 2:size(X_2d)[1]
    push!(X_1d, X_2d[i, :])
end

# kenrel
τ = 0.015
σ = 1000.
k = GaussianKernel(τ, σ)

# calculate W matrix
function least_square(y::AbstractVector, X::AbstractMatrix; λ::Float64=1e-2)
    Xᵀ = transpose(X)    
    inv(Xᵀ * X + λ .* I(size(Xᵀ)[1])) * Xᵀ * y
end

m = 20  # number of neighbors
K = kernel_matrix(k, X_1d)
W = spzeros(size(K)...)
for i in axes(K, 1)
    # find neighbors.
    ki = K[i, :]
    inds = findall(x -> x .≤ m, tiedrank(ki))
    # optimize
    W[i, inds] = least_square(X_1d[i], X_2d[inds, :]')
end

# get kernel matrix
c = 1e-8
Wᵀ = transpose(W)
K̃ = W + Wᵀ - Wᵀ * W
K_lle = K̃ + c .* I(size(K̃)[1])

# # kenrel PCA
vec_1 = ones(n_samples)
Jn = Matrix(I, n_samples, n_samples) - (vec_1 * vec_1') / n_samples
f = eigen(Jn * K_lle)
eigen_values = reverse(f.values)
eigen_vectors = reverse(f.vectors, dims=2)

function kpca(k, x_all, x, α)
    g = 0
    for i in 1:length(x_all)
        g += α[i] * kernel(k, x_all[i], x)
    end
    g
end
kpca_partial(x, α) = kpca(k, X_1d, x, α)

# # dimension reduction to 2d
ndim = 2
X_kpca = zeros(Float64, ndim, n_samples)
for j in 1:ndim
    for i in 1:n_samples
        X_kpca[j, i] = kpca_partial(X_1d[i], eigen_vectors[:, j])
    end
end

# plot
plot(X_kpca[1, inds_clusters[1]], X_kpca[2, inds_clusters[1]];
     st=:scatter, label="cluster 1", ms=5, size=(400, 400))
for i in 2:5
    plot!(
        X_kpca[1, inds_clusters[i]],
        X_kpca[2, inds_clusters[i]];
        st=:scatter,
        label=@sprintf("cluster %s", i),
        ms=5,
        size=(400, 400)
    )
end

savepath = @sprintf("%s/lle_(sigma=%s).png", @__DIR__, σ)
savefig(savepath)
