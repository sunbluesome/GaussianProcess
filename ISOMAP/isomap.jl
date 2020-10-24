using LinearAlgebra
using Arpack
using StatsBase
using LightGraphs
using SimpleWeightedGraphs
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
σ = 10.
k = GaussianKernel(τ, σ)

# adjacency matrix
m = 8
K = kernel_matrix(k, X_1d)
g = SimpleWeightedGraph(size(K)[1])
for i in axes(K, 1)
    ki = K[:, i]
    inds = findall(x -> x .≤ m, tiedrank(ki))
    for j in inds
        add_edge!(g, i, j, K[j, i])
    end
end
D =  adjacency_matrix(g)

# shortest paths by Dijkstra's algorithm
@showprogress 1 "Dijkstra's algorithm: " for i in 1:n_samples
    dijk = dijkstra_shortest_paths(g, i);
    D[:, i] .= dijk.dists
end

D_row = sum(D, dims=1) ./ n_samples
D_col = sum(D, dims=2) ./ n_samples
gm = 0.5 * (-D .+ D_row .+ D_col .- mean(D))

# kenrel PCA
vec_1 = ones(n_samples)
Jn = Matrix(I, n_samples, n_samples) - (vec_1 * vec_1') / n_samples
f = eigen(Jn * gm)
# f = eigs(gm, nev=2)
eigen_values = f.values
eigen_vectors = f.vectors

function kpca(k, x_all, x, α)
    g = 0
    for i in 1:length(x_all)
        g += α[i] * kernel(k, x_all[i], x)
    end
    g
end
kpca_partial(x, α) = kpca(k, X_1d, x, α)

# dimension reduction to 2d
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

savepath = @sprintf("%s/isomap_(sigma=%s).png", @__DIR__, σ)
savefig(savepath)
