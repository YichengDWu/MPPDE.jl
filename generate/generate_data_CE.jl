using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize
using OrdinaryDiffEq
using Distributions
using JLD2

function build_combo_eq(;
    tmin::AbstractFloat = 0.0,
    tmax::AbstractFloat = 4.0,
    xmin::AbstractFloat = 0.0,
    xmax::AbstractFloat = 16.0,
    nx::Int = 200,
)
    @parameters t x α β γ
    @parameters A[1:5] ω[1:5] ℓ[1:5] ϕ[1:5]

    @variables u(..)

    δ(t, x) = sum(
        scalarize(A) .*
        sin.(scalarize(ω) .* t .+ π / 8 .* scalarize(ℓ) .* x .+ scalarize(ϕ)),
    )

    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    Dxxx = Differential(x)^3

    eq =
        Dt(u(t, x)) ~
            -2.0 * α * u(t, x) * Dx(u(t, x)) + β * Dxx(u(t, x)) - γ * Dxxx(u(t, x)) +
            δ(t, x)

    domains = [x ∈ Interval(xmin, xmax), t ∈ Interval(tmin, tmax)]

    bcs = [u(tmin, x) ~ δ(tmin, x), u(t, xmin) ~ u(t, xmax)]

    @named combo = PDESystem(
        eq,
        bcs,
        domains,
        [t, x],
        [u(t, x)],
        vcat(
            [α => 1.0, β => 1.0, γ => 1.0],
            scalarize(A) .=> ones(5),
            scalarize(ω) .=> ones(5),
            scalarize(ℓ) .=> ones(5),
            scalarize(ϕ) .=> ones(5),
        ),
    )

    dx = (xmax - xmin) / nx
    discretization =
        MOLFiniteDifference([x => dx], t, approx_order = 4, grid_align = center_align)
    prob = discretize(combo, discretization)

    return prob
end

uniform_sample(x::AbstractFloat) = x
uniform_sample(x::Tuple) = rand(Uniform(x[1], x[2]), 1)[1]

function generate_data(
    ::Type{T} = Float32;
    ranges,
    nsamples::Int = 2096,
    tmin::AbstractFloat = 0.0,
    tmax::AbstractFloat = 4.0,
    xmin::AbstractFloat = 0.0,
    xmax::AbstractFloat = 16.0,
    nx::Int = 200,
    nt::Int = 250,
) where {T<:AbstractFloat}
    """where {T<:AbstractFloat}
    ranges: ranges for α β γ
    """
    prob = build_combo_eq(tmin = tmin, tmax = tmax, xmin = xmin, xmax = xmax, nx = nx)

    domain = [xmin xmax; tmin tmax]
    dt = (tmax - tmin) / nt
    dx = (xmax - xmin) / nx

    n_var = count(i -> typeof(i) <: Tuple, ranges)
    var_mask = [typeof(i) <: Tuple for i in ranges]
    θ = Array{T,2}(undef, (n_var, nsamples))

    u = Array{T,3}(undef, (nt + 1, nx, nsamples))

    for i = 1:nsamples
        temp = collect(map(uniform_sample, ranges))
        θ[:, i] .= temp[var_mask]

        newprob = remake(
            prob,
            p = vcat(
                temp,
                rand(Uniform(-0.5, 0.5), 5),
                rand(Uniform(-0.4, 0.4), 5),
                rand(1:3, 5),
                rand(Uniform(0, 2π), 5),
            ),
        )
        u[:, :, i] .= Array{T}(solve(newprob, Tsit5(), saveat = dt))'
    end
    return u, dx, dt, θ, domain
end

function generate_save_data(::Val{:E1})
    mkpath("datasets")

    if isfile("datasets/E1.jld2")
        println("Data already exists.")
        return
    end
    @time u, dx, dt, θ, domain = generate_data(ranges = (1.0, 0.0, 0.0))
    jldsave("datasets/E1.jld2"; u, dx, dt, θ, domain)
    println("Data saved for E1")
    return nothing
end

function generate_save_data(::Val{:E2})
    mkpath("datasets")

    if isfile("datasets/E2.jld2")
        println("Data already exists.")
        return
    end
    @time u, dx, dt, θ, domain = generate_data(ranges = (1.0, (0.0, 2.0), 0.0))
    jldsave("datasets/E2.jld2"; u, dx, dt, θ, domain)
    println("Data saved for E2")
    return nothing
end


function generate_save_data(::Val{:E3})
    mkpath("datasets")

    if isfile("datasets/E3.jld2")
        println("Data already exists.")
        return
    end
    @time u, dx, dt, θ, domain = generate_data(ranges = ((0.0, 3.0), (0.0, 4.0), (0.0, 1.0)))
    jldsave("datasets/E3.jld2"; u, dx, dt, θ, domain)
    println("Data saved for E3")
    return nothing
end

function generate_save_data(e::Symbol)
    @assert e ∈ (:E1, :E2, :E3)
    generate_save_data(Val(e))
end