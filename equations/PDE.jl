abstract type PDE end

struct CombinedEquation <: PDE
    
    α
    β
    γ
end

struct WaveEquation <: PDE
    c
end