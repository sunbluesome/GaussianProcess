include("kernel.jl")
include("gp.jl")

"""
Kernel Gradients
"""
function _grad(k::GaussianKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    K = kernel_matrix(k, x1, x2)
    dkdθ1 = K
    dkdθ2 = K .* .-(x1, x2').^2 ./ k.θ2
    return [dkdθ1, dkdθ2]
end

"""
Likelihood
"""
# function loglikelihood(gkg::Grad, xtrain::AbstractVector{T}, ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
#     -log(det(gkg.K)) - ytrain' * gkg.K_inv * ytrain
# end

# function loglikelihood_grad(gpg::Grad, Kgrad::AbstractMatrix{T}, ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
#     -tr(gpg.K_inv * Kgrad) + transpose(gpg.K_inv * ytrain) * Kgrad * (gpg.K_inv * ytrain)
# end


"""
update
"""
abstract type Grad end

mutable struct GPGrad <: Grad
    θ::Vector{Float64}
    L::Float64
end
GPGrad() = GPGrad(Float64[], -Inf)
GPGrad(θ::Vector{Real}) = GPGrad(Float64.(θ), -Inf)

function update!(gpg::GPGrad ,θ::AbstractVector{T}, L::AbstractVector{S}) where {T<:Real, S<:Real}
    gpg.θ = θ
    gpg.L = L
end
update!(gpg::GPGrad ,θ::Real, L::Real) = update!(gpg, Float64[θ], Float64(L))
update!(gpg::GPGrad ,θ::AbstractVector{Real}, L::Real) = update!(gpg, Float64.(θ), Float64(L))


function grad(gp::GaussianProcess, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    N = length(x1)
    kgrads = _grad(gp.k, x1, x2)
    noise_grad = diagm(gp.η * ones(N))
    return [kgrads..., noise_grad]
end
grad(gp::GaussianProcess, x::AbstractVector{T}) where {T<:Real} = grad(gp, x, x)

function update!(gpg::GPGrad, gp::GaussianProcess, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T<:Real, S<:Real}
    gpg.θ = grad(gp, x1, x2)
end

