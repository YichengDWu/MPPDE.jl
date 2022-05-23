using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize
using OrdinaryDiffEq
using Distributions

function build_combo_eq()
       @parameters t x α β γ
       @parameters A[1:5] ω[1:5] ℓ[1:5] ϕ[1:5]

       @variables u(..) 

       δ(t,x) = sum(scalarize(A) .* sin.(scalarize(ω) .* t .+ π/8 .* scalarize(ℓ) .* x .+ scalarize(ϕ)))

       Dt = Differential(t)
       Dx = Differential(x)
       Dxx = Differential(x)^2
       Dxxx = Differential(x)^3

       eq = Dt(u(t, x)) ~  -2. * α * u(t, x) * Dx(u(t, x)) + β * Dxx(u(t, x)) - γ * Dxxx(u(t, x)) + δ(t,x)

       tmin,tmax = 0., 4.
       xmin,xmax = 0., 16.

       domains = [x ∈ Interval(xmin, xmax),
                     t ∈ Interval(tmin, tmax)]

       bcs = [u(tmin,x) ~ δ(tmin,x),
              u(t,xmin) ~ u(t,xmax)]
              
       @named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], vcat([α=>1., β=>0., γ=>0.],
                                                                      scalarize(A).=>ones(5),
                                                                      scalarize(ω).=>ones(5),
                                                                      scalarize(ℓ).=>ones(5), 
                                                                      scalarize(ϕ).=>ones(5)))
       
       discretization = MOLFiniteDifference([x=>0.2], t, approx_order=4, grid_align=center_align)
       prob = discretize(combeq, discretization)

       return prob
end

function BurgersEq(prob::ODEProblem, η::AbstractFloat = 0.)
       @parameters α β γ
       remake(prob, p=[α=>1., β=>η, γ=>0.])
end

function KdVEq(prob::ODEProblem)
       @parameters α β γ
       remake(prob, p=[α=>3., β=>0., γ=>1.])
end

function MixedEq(prob::ODEProblem, αval::AbstractFloat, βval::AbstractFloat, γval::AbstractFloat)
       @parameters α β γ
       remake(prob, p=[α=>αval, β=>βval, γ=>γval])
end

function generate_data(prob::ODEProblem, num_samples::Int = 2096)
       @parameters A[1:5] ω[1:5] ℓ[1:5] ϕ[1:5]
       sols = []
       for i in 1:num_samples
              newprob = remake(prob, p = vcat(scalarize(A).=> rand(Uniform(-0.5, 0.5),5),
                                              scalarize(ω).=> rand(Uniform(-0.4, 0.4),5),
                                              scalarize(ℓ).=> rand(1:3,5), 
                                              scalarize(ϕ).=> rand(Uniform(0, 2π),5))) 
              solve(newprob,Tsit5(),saveat=1)
              #push(sols, solve(newprob, Tsit5()))
       end
       return sols
end

