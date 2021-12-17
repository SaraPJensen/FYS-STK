# Code of Project 3

This repository contains our code for project 3 of FYK-STK fall 2021. This README deatils the code, including how to run it.

## Structure
In the rapport, we studied two different methods for solving differential equations. The code reflects this bi-nature. The files implementing and using the genetic algorithm all start with the prefix `genetic`, while the ones dealing with a neural network start with `NN`.

## Running code
Run the desired files with python. The desired variables are sometimes hard-coded so must be changed manually.

### Dependencies
Several libraries are required to run the code. Most of these are common libraries in any project. Notable exceptions are
- Autograd
- Plotly
- tqdm

## Code Overview

### [genetic.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic.py)
Containes class for population and class for chromosomes, and a main() function which calls on the necessary functions, classes and writes the results to a file. Chosen reproduction methods and other variables are hard-coded, so must be changed manually in main() if so desired. 

### [genetic_ODE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_ODE.py)
Same functionalities and structure as genetic.py as described above, but solves two different ordinary differential equations. To choose which, the terms must be commented in.

### [genetic_PDE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_PDE.py)
Same functionalities and structure as genetic.py, but solves a different partial differential equation.

### [genetic_plot.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_plot.py)
Plotting code for results from the genetic algorithm for the Diffusion equation.

### [genetic_plot_ODE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_plot.py)
Plotting code for results from the genetic algorithm for the ODE and PDE used to compare with Lagaris et al.

### [NN_PDE_solver.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_PDE_solver.py)
Contains class for feed forward neural network and utility classes, one for activation functions, and an optimiser for gradient descent. Neural network class contains all necessary methods, with exception of diff_eq, which must be added though inheritance. Network can be saved and reloaded.

### [NN_diffusion_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_diffusion_eq.py)
Sets up 1D diffusion PDE solver by inheriance from neural network, reimplementing the necessary methods. Main function makes an instance of this to solve the PDE

### [NN_learning_schedules.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_learning_schedules.py)
Tests different learning schedules on PDE class from `NN_diffusion_eq.py`.

### [FE__diffusion_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/FE__diffusion_eq.py)
Contains class for explicit forward euler solver of 1D diffusion PDE.

### [NN_method_comparison.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_method_comparison.py)
Compares the solution for the 1D diffusin PDE from the neural network and the forward euler solver with the analytical solution. 

### [NN_sym_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_sym_eq.py)
Uses PDE_solver_NN_base to find eigenvalue of symetric matrix by solving differential equation.

### [NN_sym_count.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_sym_count.py)
Counts the rank of the eigenvalue found in the ordered set of eigenvalues for matrix A.

