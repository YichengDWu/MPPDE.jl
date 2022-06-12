function get_data(args)
    e = args.experiment
    !isfile("datasets/$e.jld2") && generate_save_data(e) #TODO: add wave equation in the future  

    @unpack domain, u, dx, dt, θ = load("datasets/$e.jld2")

    if e == :E2
        θ = θ .* 5
    elseif e == :E3
        θ = θ./[3,eltype(θ)(0.4),1]
    end

    u_train, u_test = u[:, :, 1:floor(Int, end * 0.8)], u[:, :, ceil(Int, end * 0.8):end]
    θ_train, θ_test = θ[:, 1:floor(Int, end * 0.8)], θ[:, ceil(Int, end * 0.8):end]

    x = collect(eltype(u), domain[1, 1]:dx:domain[1, 2])[2:end]
    x = reshape(x,1,length(x))
    Nx = size(x, 2)

    t = collect(eltype(u),domain[2, 1]:dt:domain[2, 2])
    t = repeat(t, 1, Nx)

    x = x ./ (domain[1, 2] - domain[1, 1]) #normalisation
    t = t ./ (domain[2, 2] - domain[2, 1])
    dt = dt / (domain[2, 2] - domain[2, 1])

    g_train = batch([
        GNNGraph(
            knn_graph(x, args.neighbors),
            ndata = (u = u_train[:, :, i], x = x, t = t, θ = repeat(θ_train[:, i], 1, Nx)),
        ) for i = 1:size(u_train, 3)
    ])

    g_test = batch([
        GNNGraph(
            knn_graph(x, args.neighbors),
            ndata = (u = u_test[:, :, i], x = x, t = t, θ = repeat(θ_test[:, i], 1, Nx)),
        ) for i = 1:size(u_test, 3)
    ])

    train_loader = DataLoader(g_train, batchsize = args.batchsize, shuffle = true)
    test_loader = DataLoader(g_test, batchsize = args.batchsize, shuffle = true)

    global dt = eltype(u)(dt)
    return train_loader, test_loader
end

function sample_single_graph(g::GNNGraph,k::Int,K::Int,N::Int)
    @unpack u,x,t,θ = g.ndata
    @views new_u = u[k-K:k-1,:]
    @views t = t[[k-1],:]
    @views target = u[k+N*K:k+(N+1)*K-1,:] 
    return GNNGraph(g,ndata = (u = new_u, x = x, t = t, θ = θ)), target
end


function sample_batched_graph(g, args, N)
    T = size(g.ndata.t, 1) # available time steps
    K = args.K

    graphs = Vector{GNNGraph}()
    target = similar(g.ndata.u,K, 0)
    for g in unbatch(g)
        g_sampled, target_sampled = sample_single_graph(g, rand(K+1:T+1-(N+1)*K), K, N)
        push!(graphs, g_sampled)
        target = hcat(target, target_sampled)
    end
    return batch(graphs), target, N 
end