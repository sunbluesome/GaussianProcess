using LinearAlgebra
using StatsBase
using PyPlot
include("../src/kernel.jl")

# samples
n_samples = 25
function make_sample(x)
    x.^2
end
X_2d = [range(-1, stop=1, length=n_samples) randn(MersenneTwister(0), n_samples)]
y_2d = [make_sample.(X_2d[:, 1]) randn(MersenneTwister(20201111), n_samples)]
labels = repeat([1, 2, 3, 4, 5], inner=(5, 1))

X_1d = [X_2d[i, :] for i in 1:n_samples]
y_1d = [y_2d[i, :] for i in 1:n_samples]
# kenrel
τ = 1.
σ = 0.08
k = GaussianKernel(τ, σ)

Kx = kernel_matrix(k, X_1d)
Ky = kernel_matrix(k, y_1d)
O = zero(Kx)
vec_1 = ones(n_samples)
Jn = Matrix(I, n_samples, n_samples) - (vec_1 * vec_1') / n_samples
ζx = 1e-5
ζy = 1e-5

P = [
    O              Kx * Jn * Ky;
    Ky * Jn * Kx   O
]
Λ = [
    Kx * Jn * Kx + ζx * Kx  O;
    O                       Ky * Jn * Ky + ζy * Ky
]

# Generalized SVD
f = eigen(P, Λ)
eigen_values = reverse(f.values)
eigen_vectors = reverse(f.vectors, dims=1)
kx(x) = kernel_matrix(k, X_1d, [x])
ky(y) = kernel_matrix(k, y_1d, [y])

# mapping into RKHS
X_map = vec(transpose(eigen_vectors[1:n_samples, 1]) * Kx)
y_map = vec(transpose(eigen_vectors[1:n_samples, 1]) * Ky)

# plot
inds = findall(x -> x == 1, labels)
fig, axes = subplots(1, 2, figsize=(8, 4))
for l in unique(labels)
    inds = findall(x -> x == l, labels)
    name = @sprintf("cluster %s", l)
    axes[1].scatter(X_2d[inds, 1], y_2d[inds, 1], label=name, s=10)
    axes[2].scatter(X_map[inds], y_map[inds], label=name, s=10)
end
axes[1].legend()
axes[2].legend()
tight_layout()

savepath = @sprintf("%s/kcca_(sigma=%s).png", @__DIR__, σ)
savefig(savepath)