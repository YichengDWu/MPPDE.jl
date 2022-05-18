using ApproxFun, LinearAlgebra, Plots


function we_dirichlet(c, t, x, s)
    """
    s: center of the Gaussian pulse
    """
    dx = Domain(x[1] .. x[2])
    dt = Domain(t[1] .. t[2])
    d = dx × dt
    Dx = Derivative(d, [1, 0])
    Dt = Derivative(d, [0, 1])
    # need to specify both ic and its derivative
    B = [I ⊗ ldirichlet(dt), I ⊗ lneumann(dt), ldirichlet(dx) ⊗ I, rdirichlet(dx) ⊗ I]

    u0 = Fun(x -> exp(-(x - s)^2), dx)
    uₜ0 = Fun(x -> -2 * c * (x - s), dx) * u0
    QR = qr([B; Dt^2 - c * c * Dx^2])
    u = \(QR, [u0; uₜ0; 0; 0; 0]; tolerance=1E-4)
    return u
end


function we_neumann(c, t, x, s)
    """
    s: center of the Gaussian pulse
    """
    dx = Domain(x[1] .. x[2])
    dt = Domain(t[1] .. t[2])
    d = dx × dt
    Dx = Derivative(d, [1, 0])
    Dt = Derivative(d, [0, 1])
    # need to specify both ic and its derivative
    B = [I ⊗ ldirichlet(dt), I ⊗ lneumann(dt), lneumann(dx) ⊗ I, rneumann(dx) ⊗ I]

    u0 = Fun(x -> exp(-(x - s)^2), dx)
    uₜ0 = Fun(x -> -2 * c * (x - s), dx) * u0
    QR = qr([B; Dt^2 - c * c * Dx^2])
    u = \(QR, [u0; uₜ0; 0; 0; 0]; tolerance=1E-4)
    return u
end

function we_mixed(c, t, x, s)
    """
    s: center of the Gaussian pulse
    """
    dx = Domain(x[1] .. x[2])
    dt = Domain(t[1] .. t[2])
    d = dx × dt
    Dx = Derivative(d, [1, 0])
    Dt = Derivative(d, [0, 1])
    # need to specify both ic and its derivative
    B = [I ⊗ ldirichlet(dt), I ⊗ lneumann(dt), ldirichlet(dx) ⊗ I, rneumann(dx) ⊗ I]

    u0 = Fun(x -> exp(-(x - s)^2), dx)
    uₜ0 = Fun(x -> -2 * c * (x - s), dx) * u0
    QR = qr([B; Dt^2 - c * c * Dx^2])
    u = \(QR, [u0; uₜ0; 0; 0; 0]; tolerance=1E-4)
    return u
end

using BenchmarkTools
@btime we_dirichlet(2.0, (0, 20), (-8, 8), 0.1)
@btime we_dirichlet(2.0, (0, 20), (-8, 8), 1.0)

@btime we_neumann(2.0, (0, 20), (-8, 8), 0.1)

@btime we_mixed(2.0, (0, 20), (-8, 8), 0.1)
@btime we_mixed(2.0, (0, 20), (-8, 8), 0.1)