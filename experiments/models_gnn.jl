using Lux, NNlib
using NeuralGraphPDE
#import Flux: unsqueeze

function Encoder(timewindow::Int, neqvar::Int, dhidden::Int = 128, act = swish)
    return Chain(
        Dense(timewindow + 2 + neqvar, dhidden, act),  #din=timewindow + 2 + n_eqvar
        Dense(dhidden, dhidden, act),
    )
end

function ProcessorLayer(ch::Pair{Int,Int},
                        timewindow::Int,
                        neqvar::Int = 0,
                        dhidden::Int = 128,
                        act = swish)
    din, dout = ch
    ϕ = Chain(
        Dense(2 * din + timewindow + 1 + neqvar, dhidden, act),
        Dense(dhidden, dhidden, act),
    )
    ψ = Chain(Dense(din + dhidden + neqvar, dhidden, act), Dense(dhidden, dout, act))
    MPPDEConv(ϕ, ψ)
end

function ProcessorLayer(in_chs::Int, out_chs::Int, timewindow::Int, neqvar::Int = 0, dhidden::Int = 128, act = swish)
    ProcessorLayer(in_chs => out_chs, timewindow, neqvar, dhidden, act)
end

function Processor(
    ch::Pair{Int,Int},
    timewindow::Int,
    neqvar::Int = 0,
    dhidden::Int = 128,
    nlayer::Int = 6,
)
    @assert ch.first == ch.second # current limitation
    Chain([SkipConnection(ProcessorLayer(ch, timewindow, neqvar, dhidden),+) for i = 1:nlayer])
end


function Decoder(timewindow::Int, act = swish)
    @assert timewindow ∈ (20, 25, 50)
    if timewindow == 20
        return Chain(Conv((15,), 1 => 8, act; stride = 4), Conv((10,), 8 => 1, act))
    elseif timewindow == 25
        return Chain(Conv((16,), 1 => 8, act; stride = 3), Conv((14,), 8 => 1, act))
    else
        Chain(Conv((12,), 1 => 8, act; stride = 2), Conv((10,), 8 => 1, act))
    end
end
struct MPSolver{E, P, D, T<: AbstractFloat} <:
        Lux.AbstractExplicitContainerLayer{(:encoder, :processor, :decoder)}
    encoder::E
    processor::P
    decoder::D
    Δt::Vector{T}
end

"""
input: Temporal × (Spatial × N)
"""
function MPSolver(;
    dt::AbstractFloat,
    timewindow::Int = 25,
    dhidden::Int = 128,
    nlayer::Int = 6,
    neqvar::Int = 0
)
    Δt = cumsum(ones(typeof(dt), timewindow) .* dt)
    MPSolver(
        Encoder(timewindow, neqvar, dhidden),
        Processor(dhidden => dhidden, timewindow, neqvar, dhidden, nlayer),
        Decoder(timewindow),
        Δt
    )
end

function (l::MPSolver)(ndata::NamedTuple, ps::NamedTuple, st::NamedTuple)
    """
    Push u[k-K:k] to u[k:k+K]
    input:
        ndata: (u,x,t,θ)   # u is already sample to be (K, Nx * batch_size)
    """
    input = reduce(vcat, values(ndata)) #TODO: add norm
    f, st_encoder = l.encoder(input, ps.encoder, st.encoder)
    h, st_processor = l.processor(f, ps.processor, st.processor)
    d, st_decoder = l.decoder(unsqueeze(h,2), ps.decoder, st.decoder)
    d = dropdims(d;dims = 2)
    u = ndata.u[[end],:] .+ l.Δt .* d
    st = merge(st,(encoder = st_encoder, processro = st_processor, decoder = st_decoder))
    return u, st
end
