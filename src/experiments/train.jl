#CUDA.allowscalar(false)
Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    experiment::Symbol
    batchsize::Int = 16      # batch size
    use_cuda::Bool = true      # if true use cuda (if available)
    neighbors::Int = 3      # number of neighbors for the knn graph
    epochs::Int = 20          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "log/"    # results path
    K::Int = 25  # timewindow
    N::Int = 1    # number of unrollings
    infotime::Int = 20
end

function mse(ŷ, y; agg = mean)
    sqrt(agg(abs2.(ŷ .- y)))
end

function eval_loss(loader, model, ps, st, device, args)
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
            output = model((u=u_bulk, x=x, t=k, θ=θ), ps, st)
            l += loss(output, target)/steps
        end
    end
    return round(l, digits=4)
end

function draw_groudtruth(g::GNNGraph)
    u = g.ndata.u' |> cpu
    x = [g.ndata.x...] |> cpu
    p = plot(x,u,title = "Ground truth")
    return p
end

function draw_prediction(g::GNNGraph,model,args)
    g = g |> gpu #TODO:device
    @unpack u, x, t, θ = g.ndata
    T = size(t, 1)
    K = args.K
    steps = T ÷ K - 1
    output = zero(u)
    output[1:K,:] = u[1:K,:]
    for s in 1:steps
        u_bulk = u[(s-1)*K+1:s*K, :]
        k = t[[s * K], :]
        output[s*K+1:(s+1)*K, :] = model(g, (u=u_bulk, x=x, t=k, θ=θ))
    end
    pred = output' |> cpu
    x = [x...] |> cpu
    p = plot(x,pred,title = "Prediction")
    return p
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

    if args.experiment == :E1
        neqvar = 0
    elseif args.experiment == :E2
        neqvar = 1
    elseif args.experiment == :E3
        neqvar = 3
    else
        error("Experiment not found")
    end


    # load data
    train_loader, test_loader, dt = get_dataloader(args)

    # model
    model = MPSolver(dt = dt, timewindow=args.K, neqvar=neqvar)

    display(model)
    ps, st = Lux.setup(Random.default_rng(), model) |> device
    # optimizer
    opt = AdamW(args.η)
    st_opt = Optimisers.setup(opt, ps)

    # logging
    #if args.tblogger
    #    tblogger = TBLogger(args.savepath, tb_overwrite)
    #    set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
    #    @info "TensorBoard logging at \"$(args.savepath)\""
    #end

    function report(epoch)
        train_loss = eval_loss(train_loader, model, device, args)
        test_loss = eval_loss(test_loader, model, device, args)
        println("Epoch: $epoch   Train: $(train_loss)   Test: $(test_loss)")
        g_train = unbatch(first(train_loader))[2]
        g_test = unbatch(first(test_loader))[2]
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss = train_loss groundtruth = draw_groudtruth(g_train) prediction = draw_prediction(g_train,model,args)
                @info "test" loss = test_loss groundtruth = draw_groudtruth(g_test) prediction = draw_prediction(g_test,model,args)
            end
        end
    end

    # training
    #@time report(0)
    function loss_func(x, y, model, ps, st)
        return mse(model(x, ps, st)[1], y, agg=sum)
    end

    @time for epoch in 1:args.epochs
        @info "Epoch $epoch..."
        p = Progress(250)
        Nmax =  epoch ≤ args.N ? epoch - 1 : args.N
        for m in 1:250  # this in expectation has every possible starting point/sample combination of the training data
            losses = Float32[]
            for (u, x, t, θ, g) in train_loader
                N = rand(0:Nmax)   # numer of pushforward steps for each batch
                u, t, g, y = batched_sample(u, t, g, args.K, N)

                u, x, t, θ, y = (u, x, t, θ, y) .|> _flatten
                u, x, t, θ, y, g = (u, x , t, θ, y, g) .|> device

                st = updategraph(st, g)

                ChainRules.@ignore_derivatives for _ in 1:N
                    u, st = model((u = u, x = x, t = t, θ = θ), ps, st) # the pushforward trick!
                    t = t .+ dt * args.K
                end

                (l,), back = Zygote.pullback(p -> loss_func((u = u, x = x, t = t, θ = θ), y, model, p, st), ps)
                gs = back(one(l))[1]
                st_opt, ps = Optimisers.update(st_opt, ps, gs)
                push!(losses, l)
            end

            next!(p)

            @info "$m"
            if m % args.infotime == 0 || m == 1
                @info "Training Loss $(mean(losses)/args.batchsize)"
            end
        end
    end
end
