using Flux
using Flux.Data: DataLoader, batch, unbatch
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
    
    model =  precision === Float32 ? f32(model) : f64(model)
    model = model |> device  
    @info "Message Passing Solver:$(num_params(model)) parameters"
    ps = Flux.params(model)

    # loss function
    loss(ŷ, y) = mse(ŷ, y; agg = x-> mean(sum(x, dims = 1))) # sum over tim, mean over space

    # optimizer
    opt = ADAMW(args.η)

    # training
    local training_loss
    for epoch in 1:args.epochs
        @showprogress for g in train_loader
            dt = g.ndata.t[2,1]-g.ndata.t[1,1]
            global dt = precision(dt)

            g, target = construct_batched_graph(g, args) .|> device
            
            @unpack u,x,t,θ = g.ndata

            for n in 1:args.N 
                u = model(g, (u=u, x=x, t=t, θ=θ)) # the pushforward trick!
                t = t .+ dt * args.K
            end

            gs = gradient(ps) do
                output = model(g, (u=dropgrad(u), x=x, t=t, θ=θ))
                training_loss = loss(output, target)
                return training_loss
            end
            update!(opt, ps, gs)
        end
        @info "Training loos at epoch $epoch:" training_loss
    end
end
