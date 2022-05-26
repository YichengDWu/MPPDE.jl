using Flux
using GraphNeuralNetworks
using Parameters
#device = CUDA.functional() ? Flux.gpu : Flux.cpu;

function Encoder(timewindow::Int, neqvar::Int, dhidden::Int=128)
    return Chain(
        Dense(timewindow + 2 + neqvar, dhidden, swish),  #din=timewindow + 2 + n_eqvar
        Dense(dhidden, dhidden, swish)
    )
end
struct ProcessorLayer <: GNNLayer
    ϕ
    ψ
end

Flux.@functor ProcessorLayer

function ProcessorLayer(ch::Pair{Int,Int}, timewindow::Int, neqvar::Int = 0, dhidden::Int=128)
    din, dout = ch
    ϕ = Chain(
        Dense(2 * din + timewindow + 1 + neqvar, dhidden, swish),
        Dense(dhidden, dhidden, swish)
    )
    ψ = Chain(
        Dense(din + dhidden + neqvar, dhidden, swish),
        Dense(dhidden, dout, swish)
    )
    ProcessorLayer(ϕ, ψ)
end

function (p::ProcessorLayer)(g::GNNGraph, ndata::NamedTuple{(:f, :u, :pos, :θ),NTuple{4, S}}) where {S<:AbstractMatrix}
    @unpack ϕ, ψ = p
    @unpack f, θ = ndata
    
    function message(xi, xj, e)
        return ϕ(cat(xi.f, xj.f, xi.u - xj.u, xi.pos - xj.pos, xi.θ, dims=1))
    end

    m = propagate(message, g, +, xi=ndata, xj=ndata)
    update = ψ(cat(f, m, θ; dims=1))
    x = size(update)[1] == size(f)[1] ? update + f : update
    return x #TODO: add Instance Norm
end


function (p::ProcessorLayer)(g::GNNGraph)
    GNNGraph(g, ndata=(f = p(g, g.ndata), u = g.ndata.u, pos = g.ndata.pos, θ = g.ndata.θ))
end

function Decoder(timewindow::Int)
    @assert timewindow ∈ (20,25,50)
    if timewindow == 20
        return Chain(
            Conv((15,),1=>8,swish;stride=4),
            Conv((10,),8=>1,swish),
        )
    elseif timewindow == 25
        return Chain(
            Conv((16,),1=>8,swish;stride=3),
            Conv((14,),8=>1,swish),
        )
    else 
        Chain(
            Conv((12,),1=>8,swish;stride=2),
            Conv((10,),8=>1,swish),
        )
    end
end
struct MP_PDE_solver
    pde::PDESystem  #Each solver is for a specific pde
    encoder::Chain
    processor::GNNChain  #Need to look into this
    decoder::Chain
end

Flux.@functor MP_PDE_solver

function MP_PDESolver(pde::PDESystem; timewindow::Int = 25, dhidden::Int=128, nlayer::Int = 6, eqvar::NamedTuple=(;))
    """
    input: Temporal × (Spatial × N)
    """
    neqvar = length(eqvar)  #eqvar should move to pde struct
    MP_PDE_solver(
        pde,
        Encoder(timewindow, neqvar, dhidden),
        Processor(dhidden=>dhidden, timewindow, neqvar, dhidden),
        Decoder(timewindow)
    )
end

function (p::MP_PDE_solver)(g::GNNGraph, data::NamedTuple)
    """
    input:
        GNNGraph with ndata set up
        data: NamedTuple{(:u, :pos, :post, :θ),NTuple{4, S}} 
        
    For simplicity α, β, γ are all fed into NN
    """
    f = encoder(cat(u,x,t,θ))
    g.ndata.f = f
    h = processor(g)
    u = decoder(h)
end