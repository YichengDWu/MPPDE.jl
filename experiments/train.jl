using Flux
using Flux.Data: DataLoader, batch, unbatch
using Flux.Optimise: update!
using Flux.Losses: mse
using TensorBoardLogger
using Logging: with_logger
using CUDA
using Zygote: dropgrad, ignore

include("./models_gnn.jl")
include("../generate/generate_data_CE.jl")
include("utilis.jl")


Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    experiment::String
    batchsize::Int = 16      # batch size
    use_cuda::Bool = true      # if true use cuda (if available)
    neighbors::Int = 6
    epochs::Int = 20          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "log/"    # results path
    K::Int = 25  # timewindow
    N::Int = 2    # number of unrollings
    infotime::Int = 4
end


@inline num_params(model) = sum(length, Flux.params(model))

# loss function
loss(ŷ, y) = sqrt(mse(ŷ, y)) # sum over tim, mean over space

function eval_loss(loader, model, device, args)
    l = 0.0f0
    K = args.K
    for g in loader
        g = g |> device
        @unpack u, x, t, θ = g.ndata

        T = size(t, 1)
        steps = T ÷ K - 1
        for s in 1:steps
            u_bulk = u[(s-1)*K+1:s*K, :]
            k = t[[s * K], :]
            target = u[s*K+1:(s+1)*K, :]
            output = model(g, (u=u_bulk, x=x, t=k, θ=θ))
            l += loss(output, target)
        end
    end
    return round(l, digits=4)
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
    precision = eltype(eltype(train_loader.data.ndata))

    @info "Train with precision $precision"

    # model
    model = MPSolver(timewindow=args.K, neqvar=neqvar)

    model = precision === Float32 ? f32(model) : f64(model)
    model = model |> device
    @info "Message Passing Solver:$(num_params(model)) parameters"
    ps = Flux.params(model)

    # optimizer
    opt = ADAMW(args.η)

    # logging
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train_loss = eval_loss(train_loader, model, device, args)
        test_loss = eval_loss(test_loader, model, device, args)
        println("Epoch: $epoch   Train: $(train_loss)   Test: $(test_loss)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss = train_loss
                @info "test" loss = test_loss
            end
        end
    end

    # training
    ignore() do
        @time report(0)
    end
    @time for epoch in 1:args.epochs
        for g in train_loader
            g, target = construct_batched_graph(g, args) .|> device

            @unpack u, x, t, θ = g.ndata

            for n in 1:args.N
                u = model(g, (u=u, x=x, t=t, θ=θ)) # the pushforward trick!
                t = t .+ dt * args.K
            end

            gs = gradient(ps) do
                output = model(g, (u=dropgrad(u), x=x, t=t, θ=θ))
                loss(output, target)
            end
            update!(opt, ps, gs)
            if epoch % args.infotime == 0
                ignore() do
                    report(epoch)
                end
            end
        end

    end
end
