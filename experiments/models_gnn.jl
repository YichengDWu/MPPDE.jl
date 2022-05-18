using Flux
using GraphNeuralNetworks
#device = CUDA.functional() ? Flux.gpu : Flux.cpu;

function Encoder(timewindow::Int, neqvar::Int, nhidden::Int=128)
    return Chain(
        Dense(timewindow + 2 + neqvar, nhidden, swish),  #din=timewindow + 2 + n_eqvar
        Dense(nhidden, nhidden, swish)
    )
end


struct ProcessorLayer <: GNNLayer
    ϕ
    ψ
end

Flux.@functor ProcessorLayer

function ProcessorLayer(ch::Pair{Int,Int}, dhidden::Int=128; timewindow::Int, neqvar::Int)
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

function (p::ProcessorLayer)(g::GNNGraph,x::AbstractMatrix,u::AbstractMatrix,pos::AbstractVector, θ::AbstractArray)
    function message(xi, xj, ui, uj, posi, posj, θi)
        return ϕ(cat(xi, xj, ui-uj, posi-posj, θi;dims = 1))
    end
    ̄m = propagate(message, g, +; xi=x, xj=x, ui=u, uj = u, posi = pos, posj= pos, θ)
    f = ψ(cat(m, θ;dims = 1))
    x = size(x)[1] == size(f)[1] ? x + f : f
    return norm(x)
end

function (p::ProcessorLayer)(g::GNNGraph) 
    GNNGraph(g, ndata = p(g, g.ndata.x, g.ndata.u, g.ndata.pos, g.ndata.θ))
end


struct MP_PDE_solver
    Encoder::Chain
    Processor:: GNNChian
    Decoder:: Chain
end

Flux.@functor MP_PDE_solver

function MP_PDE_solver(timewindow::Int, neqvar::Int, nhidden::Int=128)
    MP_PDE_solver(
        Encoder(timewindow, neqvar, nhidden),
        Processor(Pair{nhidden, nhidden}, timewindow, neqvar),
        Decoder(Pair{nhidden, nhidden}, timewindow, neqvar)
    )
end
    
function (p::MP_PDE_solver)(x)

end