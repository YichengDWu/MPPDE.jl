using ModelingToolkit

@parameters x
@variables t u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C = 1
eq = Dtt(u(t, x)) ~ C^2 * Dxx(u(t, x))

# Initial and boundary conditions
bcs = [
    u(t, 0) ~ 0.0,# for all t > 0
    u(t, 1) ~ 0.0,# for all t > 0
    u(0, x) ~ x * (1.0 - x), #for all 0 < x < 1
    Dt(u(0, x)) ~ 0.0,
] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ (0.0, 1.0), x ∈ (0.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
