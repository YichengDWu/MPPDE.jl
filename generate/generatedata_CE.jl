using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize
using OrdinaryDiffEq
using Distributions

function build_combo_eq(;tmin::AbstractFloat = 0., 
                         tmax::AbstractFloat = 4.,
                         xmin::AbstractFloat = 0.,
                         xmax::AbstractFloat = 16.,   
                         nx::Int = 200)
       @parameters t x α β γ
       @parameters A[1:5] ω[1:5] ℓ[1:5] ϕ[1:5]

       @variables u(..) 

       δ(t,x) = sum(scalarize(A) .* sin.(scalarize(ω) .* t .+ π/8 .* scalarize(ℓ) .* x .+ scalarize(ϕ)))

       Dt = Differential(t)
       Dx = Differential(x)
       Dxx = Differential(x)^2
       Dxxx = Differential(x)^3

       eq = Dt(u(t, x)) ~  -2. * α * u(t, x) * Dx(u(t, x)) + β * Dxx(u(t, x)) - γ * Dxxx(u(t, x)) + δ(t,x)

       domains = [x ∈ Interval(xmin, xmax),
                  t ∈ Interval(tmin, tmax)]

       bcs = [u(tmin,x) ~ δ(tmin,x),
              u(t,xmin) ~ u(t,xmax)]
              
       @named combo = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], vcat([α=>1., β=>1., γ=>1.],
                                                                      scalarize(A).=>ones(5),
                                                                      scalarize(ω).=>ones(5),
                                                                      scalarize(ℓ).=>ones(5), 
                                                                      scalarize(ϕ).=>ones(5)))
       
       dx = (xmax - xmin) / nx
       discretization = MOLFiniteDifference([x=>dx], t, approx_order=4, grid_align=center_align)
       prob = discretize(combo, discretization)

       return combo, prob
end

uniform_sample(x::AbstractFloat) = x
uniform_sample(x::Tuple) = rand(Uniform(x[1], x[2]),1)[1]


function generate_data(ranges, n_samples::Int = 2096)
       """
       ranges: ranges for α β γ
       """
       pde, prob = build_combo_eq()

       n_var = count(i->typeof(i)<:Tuple,ranges)
       var_mask = [typeof(i)<:Tuple for i in ranges]
       θ = Array{Float64,2}(undef, (n_var,n_samples))
       
       sols = []   
       
       for i in 1:n_samples     
              temp = collect(map(uniform_sample, ranges))
              newprob = remake(prob, p = vcat(temp,   
                                              rand(Uniform(-0.5, 0.5),5),
                                              rand(Uniform(-0.4, 0.4),5),
                                              rand(1:3,5), 
                                              rand(Uniform(0, 2π),5))) 
              push!(sols, solve(newprob, Tsit5()))
              θ[:,i] .= temp[var_mask]
       end

       return pde, θ, sols  #TODO: use Datasets.jl, add pos 
end

