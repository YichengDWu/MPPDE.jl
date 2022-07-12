using MLUtils
using GraphNeuralNetworks

function get_data(args)
    e = args.experiment
    !isfile("datasets/$e.jld2") && generate_save_data(e) #TODO: add wave equation in the future

    @unpack domain, u, dx, dt, θ = load("datasets/$e.jld2")

    if e == :E2
        θ = θ .* 5
    elseif e == :E3
        θ = θ./[3,eltype(θ)(0.4),1]
    end

    #u_train, u_test = splitobs(u, at = 0.8, shuffle = true)
    #θ_train, θ_test = splitobs(θ, at = 0.8, shuffle = true)

    x = collect(eltype(u), domain[1, 1]:dx:domain[1, 2])[2:end]
    x = reshape(x,1,length(x))
    dx = collect(@views x[:,s]-x[:,t])

    t = collect(eltype(u), domain[2, 1]:dt:domain[2, 2])
    t = reshape(t,1,length(t))

    #x = x ./ (domain[1, 2] - domain[1, 1]) #normalisation
    #t = t ./ (domain[2, 2] - domain[2, 1])
    #dt = dt / (domain[2, 2] - domain[2, 1])

    g = knn_graph(x, args.neighbors)
    s, t = edge_index(g)

    function get_graphs(u,θ)
        gg = GNNGraph(s, t, edata = (du = collect(@views u[:,:,1][:,s]- u[:,:,1][:,t]),
                                     dx = copy(dx)),
                            gdata = (;θ = collect(θ[:,[1]])))
        graphs = [gg]

        for i in 2:size(u, 3)
            gg = GNNGraph(s, t, edata = (du = collect(@views u[:,:,i][:,s]- u[:,:,i][:,t]),
                                         dx = copy(dx)),
                                gdata = collect(θ[:,[i]]))
            push!(graphs, gg)
        end
        return graphs
    end

    graphs = get_graphs(u,θ)

    train_data, test_data = splitobs((u, x, t, θ, graphs), at = 0.8, shuffle = true)

    train_loader = DataLoader(train_data, batchsize = args.batchsize, shuffle = true)
    test_loader = DataLoader(test_data, batchsize = args.batchsize, shuffle = true)

    dt = eltype(u)(dt)
    return train_loader, test_loader, dt
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
