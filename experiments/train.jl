using Flux
using Flux.Data: DataLoader
using Flux.Optimise: update!
using Flux.Losses: mse
using ProgressMeter: @showprogress
using TensorBoardLogger
using Logging: with_logger
using CUDA

include("./models_gnn.jl")
include("../generate/generate_data_CE.jl")
include("utilis.jl")


Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    experiment::String
    batchsize::Int = 128      # batch size
    use_cuda::Bool = true      # if true use cuda (if available)
    neighbors::Int = 2
    epochs::Int = 10          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "runs/"    # results path
    K::Int = 25  # timewindow
    N::Int = 1    # number of unrollings
end

function training_loop(args)
    T = size(g.ndata.t, 1) # available time steps
    N = rand(1:args.N)   # numer of pushforward steps for each batch
    t = rand(args.K+1:T+1-(N+1)*K, args.batchsize)   # startingpoint for each instance in one batch
end

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

    # model
    model = MPSolver(timewindow=args.K, neqvar=neqvar) |> device
    @info "Message Passing Solver:$(num_params(model)) parameters"
    ps = Flux.params(model)

    # optimizer
    opt = ADAMW(args.η)

    # loss function
    loss(x, y) = mse(model(x), y; agg=sum)

    # training
    for epoch in args.epochs
        @showprogress for g in train_loader
            g = g |> device
            gs = Flux.gradient(ps) do
                loss(g, y)
            end
            update!(opt, ps, gs)
        end
    end
end
