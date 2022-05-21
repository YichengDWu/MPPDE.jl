using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize

function CombEquation()
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

       bcs = [u(0,x) ~ δ(0,x),
              u(t,xmin) ~ u(t,xmax)]
              
       @named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)],[α, β, γ, A, ω, ℓ, ϕ])
       
       return combeq
end