using ModelingToolkit

@parameters t x α β γ
@variables u(..) δ(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2
Dxxx = Differential(x)^3

eq = Dt(u(t, x)) + 2.0 * α * u(t, x) * Dx(u(t, x)) - β * Dxx(u(t, x)) + γ * Dxxx(u(t, x)) ~ 0

tmin,tmax = 0, 4
xmin,xmax = 0, 16

domains = [x ∈ (xmin, xmax),
              t ∈ (tmin, tmax)]

bcs = [u(0,x) ~ δ(0,x),
       u(t,xmin) ~ u(t,xmax)]
       
@named combeq = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
