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
Description

__Classes__:
  - Chromosome:
    - Description
    - Functions:
      - read_genes():
      - expression():
      - operator():
      - func():
      - digit():
      - boundary_diff():
      - calc_fitness():
      - __ gt __ ():
      - set_fitness():
      - get_fitness():
      - get_equation():
      - return_genes():


  - Population:
     - Description
     - Functions:
        - fitness():
        - breed_mix():
        - breed_swap():
        - breed_tournament():
        - breed_random():
        - mutate():
        - print_eqs():


__Functions__:
  - main(): Runs the code with the chosen variables and writes the results to a file in /data.


### [genetic_ODE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_ODE.py)
Same functionalities and structure as genetic.py as described above, but solves two different ordinary differential equations. To choose which, the terms must be commented in.

### [genetic_PDE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_PDE.py)
Same functionalities and structure as genetic.py, but solves a different partial differential equation.



### [genetic_plot.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_plot.py)
Plotting code for results from the genetic algorithm for the Diffusion equation.

__Functions__:
  - Plotly(): plots the surface of the inputted equation using Plotly.
  - Plot(): plots the surface fo the inputted equation using Matplotlib.
  - progress_top(): Plots the fitness value of the top chomosome as function of generations for all the different selection methods tested.
  - progress_avg10(): Plots the average fitness value of the top 10 % of the chomosomes as function of generations for all the different selection methods tested.
  - progress_avg70(): Plots the average fitness value of the top 70 % of the chomosomes as function of generations for all the different selection methods tested.
  - main(): calls on the plotting function of choice. Comment in the desired one.

### [genetic_plot_ODE.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/genetic_plot.py)
Plotting code for results from the genetic algorithm for the ODE and PDE used to compare with Lagaris et al.

__Functions__:
  - progress_top(): Plots the fitness value of the top chomosome as function of generations for the three different differential equations.
  - progress_top_random(): Plots the fitness value of the top chomosome as function of generations for two of the different differential equations, using random reproduction and tournament selection.
  - main(): calls on the plotting function of choice. Comment in the desired one.

### [NN_PDE_solver.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_PDE_solver.py)
Contains class for feed forward neural network and utility classes, one for activation functions, and an optimiser for gradient descent. Neural network class contains all nessisary methods, with exception of diff_eq, which must be added though inheritance. Network can be saved and reloaded.

### [NN_diffusion_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_diffusion_eq.py)
Sets up 1D diffusion PDE solver by inheriance from neural network, reimplementing the nessisary methods. Main funtion makes an instance of this to solve the PDE

### [NN_learning_schedules.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_learning_schedules.py)
Tests different learning schedules on PDE class from `NN_diffusion_eq.py`.

### [FE__diffusion_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/FE__diffusion_eq.py)
Contains class for explisit forward euler solver of 1D diffusion PDE.

### [NN_method_comparison.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_method_comparison.py)
Compares the solution for the 1D diffusin PDE from the neural network and the forward euler solver with the analytical solution.

### [NN_sym_eq.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_sym_eq.py)
Uses PDE_solver_NN_base to find eigenvalue of symetric matrix by solving differential equation.

### [NN_sym_count.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/NN_sym_count.py)
Counts the rank of the eigenvalue found in the ordered set of eigenvalues for matrix A.

