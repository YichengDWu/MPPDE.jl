# MP PDE Solver in Julia

This is an implementation of the ["Message Passing Neural PDE Solvers"](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers) in Julia

**Warning**: rrule of `repeat` is causing scalar indexing. Must wait for upstream fixes.

## Start Julia
Lets start Julia with mutiple threads:

```julia
$ julia --threads auto
```
## Set up environment
`cd` to the project directory and call

```julia
pkg> activate .
julia> using MPPDE
```

## Produce datasets for tasks E1, E2, E3
`generate_save_data(:E1), generate_save_data(:E2), generate_save_data(:E3:)`

## Train MP-PDE solvers for tasks E1, E2, E3
`train(experiment=:E1),train(experiment=:E2),train(experiment=:E3)`

"...training for the different experiments takes between 12 and 24 hours on average on a GeForceRTX 2080Ti GPU"

