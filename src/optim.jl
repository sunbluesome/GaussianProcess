include("kernel.jl")
include("grad.jl")

abstract type AbstractOptimize end

"""
Gradient Descent
"""
mutable struct GradientDescent <: AbstractOptimize
    params::Vector{Float64}
    lr::Float64
end
GradientDescent(params::Vector{Real}, lr::Real) = GradientDescent(Float64.(params), Float64(lr))
GradientDescent(params::Real, lr::Real) = GradientDescent(Float64[params], Float64(lr))

function step!(gd::GradientDescent, f::Function, g::Function)  
    grads = g(gd.params)
    for (i,grad) in enumerate(grads)
        gd.params[i] -= gd.lr * grad
    end
end


"""
Optimize
"""
abstract type AbstractOptimizeBase end
    
mutable struct Optimize <: AbstractOptimizeBase
    method::AbstractOptimize
    tol::Float64
end
Optimize(method::AbstractOptimize, tol::Real) = Optimize(method, Float64(tol))

function minimize!(opt::Optimize, f::Function, g::Function; verbose=100)
    tol = Inf
    val1 = f(opt.method.params)
    if verbose == 0
        log = []
    else
        log = [[opt.method.params..., val1, tol]]
    end
    
    cnt = 1
    while opt.tol < tol
        step!(opt.method, f, g)
        val2 = f(opt.method.params)
        tol = abs(val2 - val1)
        val1 = val2
        if verbose != 0 && mod(cnt, verbose)==0
            push!(log, [opt.method.params..., val1,tol])
        end
        cnt += 1
    end
        
    return (opt, [val1, tol], hcat(log...)')
end
