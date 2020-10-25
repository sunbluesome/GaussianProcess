using LinearAlgebra
using StatsBase
using PyCall
using Plots
pyplot(fmt=:png)
include("../src/kernel.jl")
datasets = pyimport("sklearn.datasets")

# swiss roll
n_samples = 1500
noise = 0.0
random_state = 0
X_2d, _ = datasets.make_swiss_roll(n_samples; noise=noise, random_state=random_state)

# standardize
# dt = fit(ZScoreTransform, X_2d, dims=1)
# StatsBase.transform!(dt, X_2d)

X_1d = [X_2d[1, :]]
for i in 2:size(X_2d)[1]
    push!(X_1d, X_2d[i, :])
end


# kenrel
τ = 0.015
σ = 10.
k = GaussianKernel(τ, σ)

# kenrel PCA
vec_1 = ones(n_samples)
Jn = Matrix(I, n_samples, n_samples) - (vec_1 * vec_1') / n_samples
K = kernel_matrix(k, X_1d)
f = eigen(Jn * K)
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

# dimension reduction to 2d
ndim = 2
X_kpca = zeros(Float64, ndim, n_samples)
for j in 1:ndim
    for i in 1:n_samples
        X_kpca[j, i] = kpca_partial(X_1d[i], eigen_vectors[:, j])
    end
end


plot(X_kpca[1, :], X_kpca[2, :];
     st=:scatter, label="kpca", ms=5, color=:blue, size=(400, 400))

savepath = @sprintf("%s/kpca_(sigma=%s).png", @__DIR__, σ)
savefig(savepath)
