# Project 3 - Solving the Diffusion Equation using Neural Networks and Genetic Algorithms


This repository contains the work of Sara Pernille Jensen and Håkon Olav Torvik for project 3 of FYS-STK - Applied Data Analysis and Machine Learning, fall 2021.

The project rapport [rapport.pdf](https://github.com/SaraPJensen/FYS-STK/blob/main/Project3/tex/rapport.pdf) is found in the tex directory.

# Solving differential equations
Machine learning algorithms have many applications, and were in this paper used to solve partial differential equations and non-linear differential equations. The methods used were a feed forward neural network and a genetic algorithm inspired by darwinian evolution. While the neural network gave good results when solving the diffusion equation, it was slow and considerably slower than explicit solvers. The genetic algorithm was able to find the solution to simple differential equations, but was unable to find it for the diffusion equation. Fundamental flaws were found in the algorithm, and ways to fix these were discussed. Finally, the neural network was used to solve a non-linear differential equation where the solutions were eigenvectors of a symmetric matrix. This can be used to solve differential equaions reformulatable as eigenvalue problems.
