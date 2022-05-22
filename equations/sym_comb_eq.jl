using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize
using OrdinaryDiffEq
using Distributions

function CombinedEq()
       @parameters t x α β γ
       @parameters A[1:5] ω[1:5] ℓ[1:5] ϕ[1:5]

       @variables u(..) 

       δ(t,x) = sum(scalarize(A) .* sin.(scalarize(ω) .* t .+ π/8 .* scalarize(ℓ) .* x .+ scalarize(ϕ)))

       Dt = Differential(t)
       Dx = Differential(x)
       Dxx = Differential(x)^2
       Dxxx = Differential(x)^3

       eq = Dt(u(t, x)) ~  -2.0 * α * u(t, x) * Dx(u(t, x)) + β * Dxx(u(t, x)) - γ * Dxxx(u(t, x)) + δ(t,x)

       tmin,tmax = 0, 4
       xmin,xmax = 0, 16

       domains = [x ∈ Interval(xmin, xmax),
                     t ∈ Interval(tmin, tmax)]

       bcs = [u(tmin,x) ~ δ(tmin,x),
              u(t,xmin) ~ u(t,xmax)]
              
       @named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],[α=>1., β=>0, γ=>0, A=>zeros(5), ω=>zeros(5), ℓ=>zeros(5), ϕ=>zeros(5)])
       
       discretization = MOLFiniteDifference([x=>0.2], t, approx_order=2, grid_align=center_align)
       prob = discretize(combeq,discretization)

       return prob
end

function BurgersEq(prob::ODEProblem)
       remake(prob, p=[α=>1., β=>0, γ=>0])
end

function generate_data(prob::ODEProblem, num_samples::Int = 2096)
       sols = []
       for _ in 1:num_samples,
              newprob = remake(prob, p = [A => rand(Uniform(-0.5, 0.5),5), 
                                          ω => rand(Uniform(-0.4, 0.4),5),
                                          ℓ => rand(1:3,5),
                                          ϕ => rand(Uniform(0, 2π),5)])  # 
              push(sols, solve(newprob, Tsit5()))
       end
       return sols
end

@parameters t x
@parameters ω[1:5]
@variables u(..) 

Dt = Differential(t)
Dx = Differential(x)

eq = Dt(u(t, x)) ~  -2.0  * Dx(u(t, x))

tmin,tmax = 0, 4
xmin,xmax = 0, 1

domains = [x ∈ Interval(xmin, xmax),
           t ∈ Interval(tmin, tmax)]

bcs = [u(0,x) ~ 0,
       u(t,xmin) ~ u(t,xmax)]
       
@named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],[ω=>ones(5)])
@named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

discretization = MOLFiniteDifference([x=>0.2], t, approx_order=2, grid_align=center_align)
prob = discretize(combeq,discretization)
