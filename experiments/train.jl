using Flux
using Flux.Data: DataLoader
using CUDA



Base.@kwdef mutable struct Args
    experiment = "E1"
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 10          # number of epochs
    tblogger = true      # log training with tensorboard
    savepath = "runs/"    # results path
end


function train(;kws...)
    device = CUDA.functional() ? Flux.gpu : Flux.cpu
end