function get_data(args)
    e = args.experiment
    !isfile("datasets/$e.jld2") && generate_save_data(e) #TODO: add wave equation in the future  

    @unpack domain, u, dx, dt, θ = load("datasets/$e.jld2")

    u_train, u_test = u[:, :, 1:floor(Int, end * 0.8)], u[:, :, ceil(Int, end * 0.8):end]
    θ_train, θ_test = θ[:, 1:floor(Int, end * 0.8)], θ[:, ceil(Int, end * 0.8):end]

    x = collect(domain[1, 1]:dx:domain[1, 2])[2:end]
    x = [x...;;]
    Nx = size(x, 2)

    t = collect(domain[2, 1]:dt:domain[2, 2])
    t = repeat(t, 1, Nx)

    g_train = Flux.batch([
        GNNGraph(
            knn_graph(x, args.neighbors),
            ndata = (u = u_train[:, :, i], x = x, t = t, θ = repeat(θ_train[:, i], 1, Nx)),
        ) for i = 1:size(u_train, 3)
    ])

    g_test = Flux.batch([
        GNNGraph(
            knn_graph(x, args.neighbors),
            ndata = (u = u_test[:, :, i], x = x, t = t, θ = repeat(θ_test[:, i], 1, Nx)),
        ) for i = 1:size(u_test, 3)
    ])

    train_loader = DataLoader(g_train, batchsize = args.batchsize, shuffle = true)
    test_loader = DataLoader(g_test, batchsize = args.batchsize, shuffle = true)

    return train_loader, test_loader
end
