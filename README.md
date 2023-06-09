# MP PDE Solver in Julia

This is an implementation of the [message passing solver](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers) in Julia

Note that training E1 would throw an error, it will be fixed from upstream latter

Loss is not decreasing much, I will debug it latter. Or it will be fixed with Lux

There are two major branches `Lux` and `Flux` corresponding to two different implementations of the neural networks.
Choose the one you prefer.

## Set up environment
`cd` to the project directory and call
```
pkg> activate .
pkg> instantiate
```

## Produce datasets for tasks E1, E2, E3
`generate_save_data(:E1), generate_save_data(:E2), generate_save_data(:E3:)`

## Train MP-PDE solvers for tasks E1, E2, E3
`train(experiment=:E1),train(experiment=:E2),train(experiment=:E3)`

"...training for the different experiments takes between 12 and 24 hours on average on a GeForceRTX 2080Ti GPU"

