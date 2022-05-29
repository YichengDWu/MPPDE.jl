using Flux
using Flux.Data: DataLoader
using Flux.Optimise: update!
using Flux.Losses: mse
using ProgressMeter: @showprogress
using TensorBoardLogger
using Logging: with_logger
using CUDA
using Zygote: dropgrad

include("./models_gnn.jl")
include("../generate/generate_data_CE.jl")
include("utilis.jl")


Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    experiment::String
    batchsize::Int = 16      # batch size
    use_cuda::Bool = true      # if true use cuda (if available)
    neighbors::Int = 2
    epochs::Int = 10          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "runs/"    # results path
    K::Int = 25  # timewindow
    N::Int = 1    # number of unrollings
end

@inline function construct_graph(g, args)
    T = size(g.ndata.t, 1) # available time steps
    N = rand(1:args.N)   # numer of pushforward steps for each batch
    K = args.K
    tid = rand(K+1:T+1-(N+1)*K, args.batchsize)   # starting point for each instance in one batch
    
    num_nodes = size(g.ndata.u, 2)
    Nx = Flux.unbatch(g)[1].num_nodes

    tspan = g.ndata.t[:,1]
    u =  Matrix{eltype(g.ndata.u)}(undef,(K,num_nodes))
    t = Matrix{eltype(g.ndata.t)}(undef,(1,num_nodes))
    target = Matrix{eltype(g.ndata.u)}(undef,(K,num_nodes))
    for i in 1:args.batchsize
        u[:,(i-1)*Nx+1:i*Nx] .= g.ndata.u[tid[i]-K:tid[i]-1,(i-1)*Nx+1:i*Nx]
        t[:,(i-1)*Nx+1:i*Nx] .= tspan[tid[i]-1]
        target[:,(i-1)*Nx+1:i*Nx] .= g.ndata.u[tid[i]+N*K:tid[i]+(N+1)*K-1,(i-1)*Nx+1:i*Nx]
    end

    x = g.ndata.x
    θ = g.ndata.θ
    return GNNGraph(g,ndata=(;)), u, x, t, θ, target
end

@inline num_params(model) = sum(length, Flux.params(model)) 

function train(; kws...)
    args = Args(; kws...)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    if args.experiment == "E1"
        neqvar = 0
    elseif args.experiment == "E2"
        neqvar = 1
    elseif args.experiment == "E3"
        neqvar = 3
    else
        error("Experiment not found")
    end


    # load data
    train_loader, test_loader = get_data(args)
    @info "Dataset $(args.experiment): $(train_loader.nobs) train and $(test_loader.nobs) test examples"
    precision = eltype(eltype(train_loader.data.ndata))
    
    @info "Train with precision $precision"

    # model
    model = MPSolver(timewindow=args.K, neqvar=neqvar)
    
    model =  precision == Float32 ? f32(model) : f64(model)
    model = model |> device  
    @info "Message Passing Solver:$(num_params(model)) parameters"
    ps = Flux.params(model)

    # optimizer
    opt = ADAMW(args.η)

    # training
    local training_loss
    for epoch in args.epochs
        @showprogress for g in train_loader
            dt = g.ndata.t[2,1]-g.ndata.t[1,1]
            dt = precision(dt)
            g, u, x, t, θ, target = construct_graph(g, args) .|> device
            
            for n in 1:args.N
                 u = model(g, (u=u, x=x, t=t .+ dt * args.K * (n - 1), θ=θ)) # the pushforward trick!
                 u = dropdims(u;dims = 2)
            end

            gs = gradient(ps) do
                output = model(g, (u=dropgrad(u), x=x, t=t, θ=θ))
                output = dropdims(output;dims=2)
                training_loss = mse(output, target)
                return training_loss
            end
            update!(opt, ps, gs)
            @info "Training loos at epoch $epoch:" training_loss
        end
    end
end
