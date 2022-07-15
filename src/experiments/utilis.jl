function get_dataloader(args)
    e = args.experiment
    !isfile("datasets/$e.jld2") && generate_save_data(e) #TODO: add wave equation in the future

    (;domain, u, dx, dt, θ) = load("datasets/$e.jld2")

    u = u[:,1:2:end,:] # downsample
    dx = dx * 2

    if e == :E2
        θ = θ .* 5
    elseif e == :E3
        θ = θ./[3,0.4f0,1]
    end

    #u_train, u_test = splitobs(u, at = 0.8, shuffle = true)
    #θ_train, θ_test = splitobs(θ, at = 0.8, shuffle = true)

    x = collect(eltype(u), domain[1, 1]:dx:domain[1, 2])[2:end]
    x = reshape(x,1,length(x))

    g = knn_graph(x, args.neighbors)
    src, target = edge_index(g)
    dx = collect(@views x[:,src]-x[:,target])

    x = repeat(x, 1, 1, size(u,3))

    t = collect(eltype(u), domain[2, 1]:dt:domain[2, 2])
    t = repeat(t, 1, size(u,2), size(u,3))

    #x = x ./ (domain[1, 2] - domain[1, 1]) #normalisation
    #t = t ./ (domain[2, 2] - domain[2, 1])
    #dt = dt / (domain[2, 2] - domain[2, 1])

    function get_graphs(u,θ)
        if e == :E1
            gg = GNNGraph(g, edata = (du = collect(@views u[:,:,1][:,src]- u[:,:,1][:,target]),
                         dx = copy(dx)))
        else
            gg = GNNGraph(g, edata = (du = collect(@views u[:,:,1][:,src]- u[:,:,1][:,target]),
                                      dx = copy(dx)),
                             gdata = (;θ = collect(θ[:,[1]])))
        end
        graphs = [gg]

        for i in 2:size(u, 3)
            if e == :E1
                gg = GNNGraph(g, edata = (du = collect(@views u[:,:,i][:,src]- u[:,:,i][:,target]),
                                          dx = copy(dx)))
            else
                gg = GNNGraph(g, edata = (du = collect(@views u[:,:,i][:,src]- u[:,:,i][:,target]),
                                          dx = copy(dx)),
                                 gdata = (;θ = collect(θ[:,[i]])))
            end
            push!(graphs, gg)
        end
        return graphs
    end

    graphs = get_graphs(u,θ)

    θ = reshape(repeat(θ, g.num_nodes), size(θ, 1), g.num_nodes, size(u,3))
    train_data, test_data = splitobs((u, x, t, θ, graphs), at = 0.9, shuffle = true)

    train_loader = DataLoader(train_data, batchsize = args.batchsize, shuffle = true, parallel = true)
    test_loader = DataLoader(test_data, batchsize = args.batchsize, shuffle = false, parallel = true)

    dt = eltype(u)(dt)
    return train_loader, test_loader, dt
end

function single_sample(u::AbstractMatrix, t::AbstractMatrix, g::GNNGraph,k::Int,K::Int,N::Int)
    @views new_u = u[k-K:k-1,:]
    @views new_t = t[[k-1],:]
    @views target = u[k+N*K:k+(N+1)*K-1,:]

    @views new_du = g.edata.du[k-K:k-1,:]

    gdata = merge(g.gdata, (;t = new_t[:,[1]]))
    new_g = GNNGraph(g, edata = (du = new_du, dx = g.edata.dx), gdata = gdata)

    return new_u, new_t, new_g, target
end

function batched_sample(u::AbstractArray, t::AbstractArray, graphs::Vector{<:GNNGraph}, K::Int, N::Int)
    T = size(t, 1) # available time steps
    range = K+1:T+1-(N+1)*K

    Nx = size(u, 2) # number of nodes
    nsamples = length(graphs)

    sampled_u = similar(u, K, Nx, nsamples)
    sampled_t = similar(t, 1, Nx, nsamples)
    sampled_graphs = similar(graphs)
    sampled_targets = similar(u, K, Nx, nsamples)

    for i in 1:nsamples
        u_, t_, g_, target_ = single_sample(u[:,:,i], t[:,:,i], graphs[i], rand(range), K, N)
        sampled_u[:, :, i] .= u_
        sampled_t[:, :, i] .= t_
        sampled_graphs[i] = g_
        sampled_targets[:, :, i] .= target_
    end
    return sampled_u, sampled_t, batch(sampled_graphs), sampled_targets
end

@inline function _flatten(x::AbstractArray)
    s = size(x)
    reshape(x, s[1:(end-2)]..., s[end-1]*s[end])
end
