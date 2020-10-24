using LinearAlgebra
using StatsBase
using PyCall
using Plots
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

# Lhaplacian Eigen Map
K = kernel_matrix(k, X_1d)
Λ = diagm(vec(sum(K, dims=1)))
P = Λ - K
f = eigen(P, Λ)
eigen_values = f.values
eigen_vectors = f.vectors

X_lem = eigen_vectors[:, 2:3]

# plot
plot(X_lem[inds_clusters[1], 1], X_lem[inds_clusters[1], 2];
     st=:scatter, label="cluster 1", ms=5, size=(400, 400))
for i in 2:5
    plot!(
        X_lem[inds_clusters[i], 1],
        X_lem[inds_clusters[i], 2];
        st=:scatter,
        label=@sprintf("cluster %s", i),
        ms=5,
        size=(400, 400)
    )
end

savepath = @sprintf("%s/lem_(sigma=%s).png", @__DIR__, σ)
savefig(savepath)