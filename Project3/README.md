# Project 3 - Solving the Diffusion Equation using Neural Networks and Genetic Algorithms

This repository contains our code for project 3 of FYK-STK fall 2021. This README deatils the code, including how to run it.

## Running code
Run the desired files. The desired variables are sometimes hard-coded so must be changed manually. 

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
