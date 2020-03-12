include("kernel.jl")


"""
Multivariate normal distribution
"""
function mvnorm(μ::AbstractVector{T}, K::AbstractMatrix{W}; seed::Integer=-1) where {T<:Real, W<:Real}
    U,S,V = svd(K)
    rng = MersenneTwister()
    if seed >= 0 
        rng = MersenneTwister(seed)
    end
    x = randn(rng,size(K)[1])
    y = μ .+ U*Diagonal(sqrt.(S))*x
end

"""
random sample by GP
"""
function sample(k::Kernel, xx::AbstractVector; seed::Integer=-1)
    K = kernel_matrix(k, xx)
    N = size(xx)[end]
    mvnorm(zeros(N), K; seed=seed) 
end


"""
Gaussian Process
"""
abstract type PredictMethod end

mutable struct SubsetOfData <: PredictMethod 
    indices::Vector{Int64}
end 
SubsetOfData(indices::Vector{Real}) = SubsetOfData(Int64.(indices))
SubsetOfData(indices::Real) = SubsetOfData(Int64[indices])
SubsetOfData() = SubsetOfData(Int64[])

mutable struct FITC <: PredictMethod 
    iv::Vector{Float64}
end 
FITC(iv::Vector{Real}) = FITC(Int64.(iv))
FITC(iv::Real) = FITC(Int64[iv])



mutable struct GaussianProcess 
    k::Kernel
    η::Float64
    function GaussianProcess(k::Kernel, η::Float64)
        new(k, η)
    end
end
GaussianProcess(k::Kernel) = GaussianProcess(k,0.0)
GaussianProcess(k::Kernel, η::Real) = GaussianProcess(k,Float64(η))

function logderiv(gp::GaussianProcess, x1::AbstractVector, x2::AbstractVector)
    klogderiv = logderiv(gp.k, x1, x2)
    nlogderiv = diagm(gp.η*ones(length(x1)))
    return [klogderiv..., nlogderiv]
end
logderiv(gp::GaussianProcess, xx::AbstractVector) = logderiv(gp, xx, xx)

function _predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
                  ytrain::AbstractVector{T}, Kinv::AbstractMatrix{S}, method::SubsetOfData) where {T<:Real, S<:Real}
    s = kernel_matrix(gp.k, xtest) + [gp.η]
    k = kernel_matrix(gp.k, xtrain, xtest)
    μ = k'*Kinv*ytrain
    σ2 = s - k'*Kinv*k
    return μ, σ2
end
_predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector, ytrain::AbstractVector{T},
         Kinv::AbstractMatrix{S}, method::SubsetOfData) where {T<:Real, S<:Real} = _predict(gp, [xtest], xtrain, ytrain, Kinv, method)

function predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, method::SubsetOfData) where {T<:Real}
    μs = zero(xtest)
    σs = zero(xtest)
    xtrain_reduce = length(method.indices)==0 ? xtrain : xtrain[method.indices]
    ytrain_reduce = length(method.indices)==0 ? ytrain : ytrain[method.indices]

    K = kernel_matrix(gp.k, xtrain_reduce) + diagm(gp.η*ones(length(xtrain_reduce)))
    Kinv = Symmetric(inv(bunchkaufman(K)))
    for (i,x) in enumerate(xtest)
        _μ, _σ = _predict(gp, x, xtrain_reduce, ytrain_reduce, Kinv, method)
        μs[i], σs[i] = _μ[1], _σ[1]
    end
    σs[findall(x->x<0, σs)] .= 0.;
    return μs, σs
end
predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real} = predict(gp, xtest, xtrain, ytrain, SubsetOfData())
predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, method::SubsetOfData) where {T<:Real} = predict(gp, [xtest], xtrain, ytrain, method)
predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}) where {T<:Real} = predict(gp, [xtest], xtrain, ytrain, SubsetOfData())



# function _predict(gp::GaussianProcess, xt::AbstractVector, u::AbstractVector, ytrain::AbstractVector{T}, 
#         Σ::AbstractMatrix{S}, Kuf::AbstractMatrix{S}, Λi::AbstractMatrix{S}, method::FITC) where {T<:Real, S<:Real}
        
#     Ktu = kernel_matrix(gp.k, xt, u)
#     Ktt = kernel_matrix(gp.k, xt)

#     μ = Ktu * Σ * Kuf * Λi * ytrain
#     σ2 = Ktt
#     return μ, σ2
# end
# _predict(gp::GaussianProcess, xt::Real, u::AbstractVector, ytrain::AbstractVector{T},
#          Σ::AbstractMatrix{S}, Kuf::AbstractMatrix{S}, Λi::AbstractMatrix{S}, method::FITC) where {T<:Real, S<:Real} = _predict(gp, [xt], u, ytrain, Kut, Σ, Kuf, Λi, method)

function predict(gp::GaussianProcess, xtest::AbstractVector, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, method::FITC) where {T<:Real}
    """
    Inducing Variable Methodではp(u)→p(f|u)→p(y|f)と3段階の生成過程を経ると考える。  
    データのノイズはp(y|f)で乗ってくるので、u, fを扱っている間はノイズが入ってくることはない。
    """
    u = method.iv
    N = length(xtrain)
    M = length(xtest)

    # get p(u|y)
    Kff = Diagonal([kernel(gp.k, x, x) for x in xtrain])
    Kuf = kernel_matrix(gp.k, u, xtrain)
    Kuu = kernel_matrix(gp.k, u)
    Kuui = Symmetric(inv(Kuu))
    Kut = kernel_matrix(gp.k, u, xtest)
    Ktt = kernel_matrix(gp.k, xtest)

    Qfu = Kuf' * Kuui
    Qff = Diagonal([dot(Qfu[i,:], Kuf[:,i]) for i in 1:N])
    Qtu = Kut' * Kuui
    Qtt = Diagonal([dot(Qtu[i,:], Kut[:,i]) for i in 1:M])

    Λ = Kff - Qff + Diagonal(gp.η*ones(length(xtrain)))
    Λi = Symmetric(inv(Λ)) 
    Σ = Symmetric(inv(Kuu + Kuf * Λi * Kuf'))

    μs = Kut' * Σ * Kuf * Λi * ytrain
    σs = Ktt - Qtt + Diagonal(gp.η*ones(length(xtest)))
    σs[findall(x->x<0, σs)] .= 0.;
    return μs, σs
end

predict(gp::GaussianProcess, xtest::Real, xtrain::AbstractVector,
        ytrain::AbstractVector{T}, method::FITC) where {T<:Real} = predict(gp, [xtest], xtrain, ytrain, method)

function update!(gp::GaussianProcess, params::Vector{T}) where {T<:Real}
    update!(gp.k, params[1:end-1])
    gp.η = params[end]
end


"""
Likelihood
"""
function loglikelihood(gp::GaussianProcess, xtrain::AbstractVector{T},
        ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    N = length(xtrain)
    K = kernel_matrix(gp.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
    Kinv = Symmetric(inv(K))
    -logdet(K) - ytrain' * Kinv * ytrain
end

function loglikelihood_partialderiv(gp::GaussianProcess, ytrain::AbstractVector{S},
        Kinv::Matrix{U}, Kgrad::Matrix{U}) where {T<:Real, S<:Real, U<:Real}
    -tr(Kinv * Kgrad) + transpose(Kinv * ytrain) * Kgrad * (Kinv * ytrain)
end

function loglikelihood_deriv(gp::GaussianProcess, xtrain::AbstractVector{T},
        ytrain::AbstractVector{S}) where {T<:Real, S<:Real}
    K = kernel_matrix(gp.k, xtrain) + diagm(gp.η*ones(length(xtrain)))
    Kinv = Matrix(Symmetric(inv(K)))
    Kinvy = Kinv * ytrain
    Kgrads = logderiv(gp, xtrain)
    [loglikelihood_partialderiv(gp, ytrain, Kinv, Kgrad) for Kgrad in Kgrads]
end
