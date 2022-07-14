module MPPDE

using Lux, Random
using GraphNeuralNetworks
using Optimisers
using TensorBoardLogger
using Logging: with_logger
using CUDA
using Zygote, ChainRules
using Plots
using Statistics: mean
using ProgressMeter
using MLUtils
using GraphNeuralNetworks
using Parameters
using JLD2
using Lux, NNlib
using NeuralGraphPDE
using ModelingToolkit, MethodOfLines, DomainSets
using Symbolics: scalarize
using OrdinaryDiffEq
using Distributions
using JLD2

include("./experiments/train.jl")
include("./experiments/utilis.jl")
include("./experiments/models_gnn.jl")
include("./generate/generate_data_CE.jl")

export train, generate_data_CE

end
