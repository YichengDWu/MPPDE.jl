using Flux
using Flux.Data: DataLoader
using Flux.Optimise: update!
using ProgressMeter: @showprogress
using CUDA

include("./models_gnn.jl")
include("../generate/generate_data_CE.jl")
include("utilis.jl")
Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    experiment::String
    timewindow::Int = 25
    batchsize::Int = 128      # batch size
    use_cuda::Bool = true      # if true use cuda (if available)
    neighbors::Int = 2
    epochs::Int = 10          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "runs/"    # results path
end

function train(;kws...)
    args = Args(;kws...)
    use_cuda = args.use_cuda && CUDA.functional()
    
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load data
    train_loader, test_loader = get_data(args)
    @info "Dataset $(args.experiment): $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    # model
    model = MPSolver() |> device
    @info "Message Passing Solver:$(num_params(model)) parameters"
    ps = Flux.params(model)  

    # optimizer
    opt = ADAMW(args.η)

    # loss function

    # training
    for epoch in args.epochs
        @showprogress for g in train_loader
            g = g |> device
            gs = Flux.gradient(ps) do 
                    ŷ = model(g)
                    loss(ŷ,y)
                end
            update!(opt,ps,gs)
        end
    end
end