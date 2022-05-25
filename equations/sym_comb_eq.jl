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
              
       @named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], vcat([α=>1., β=>0., γ=>0.],
                                                                      scalarize(A).=>ones(5),
                                                                      scalarize(ω).=>ones(5),
                                                                      scalarize(ℓ).=>ones(5), 
                                                                      scalarize(ϕ).=>ones(5)))
       
       dx = (xmax - xmin) / nx
       discretization = MOLFiniteDifference([x=>dx], t, approx_order=4, grid_align=center_align)
       prob = discretize(combeq, discretization)

       return prob
end

function BurgersEq(prob::ODEProblem, η::AbstractFloat = 0.)
       p = prob.p
       p[1:3] .= 1., η, 0.
       remake(prob, p=p)
end

function KdVEq(prob::ODEProblem)
       p = prob.p
       p[1:3] .= 3., 0., 1.
       remake(prob, p=p)
end

function MixedEq(prob::ODEProblem, α::AbstractFloat, β::AbstractFloat, γ::AbstractFloat)
       p = prob.p
       p[1:3] .= α, β, γ
       remake(prob, p=p)
end

function generate_data(prob::ODEProblem, num_samples::Int = 2096)
       sols = []
       for i in 1:num_samples
              newprob = remake(prob, p = vcat(prob.p[1:3],
                                              rand(Uniform(-0.5, 0.5),5),
                                              rand(Uniform(-0.4, 0.4),5),
                                              rand(1:3,5), 
                                              rand(Uniform(0, 2π),5))) 
              #solve(newprob,Tsit5(),saveat=1)
              push!(sols, solve(newprob, Tsit5()))
       end
       return sols
end

