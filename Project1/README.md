# FYS-STK

## Regression analysis and resampling methods.

Requirements
```
numpy
matplotlib
plotly
scikit-learn
imageio
time
mpl_toolkits.mplot3d
random
```

Functions for regression methods and other is contained in the file Functions.py. Scalers.py contains functions for the different kinds of scaling that is done the data.
The files exercises.py contain if/else block for generating data and calling appropriate functions, corresponding to the problem in the project text (found [here](https://github.com/CompPhysics/MachineLearning/tree/master/doc/Projects/2021/Project1)). The main block need to be calling with appropriate parameters:

Exercises 1-5 can be ran by calling
```
main(x, writeData)
```
where x is an integer corresponding to the exercise, and writeData is a boolean value used by exercise 4 and 5. Exercise 4 and 5 will make datafiles if writeData is set to true. This data can be plotted by calling
```
main("plot", False)
```
Exercise 6 can be executed by calling the function terrain.
```
terrain(part)
```
where part is a string which can take the following values: "OLS_tradeoff", "Ridge_lambda", "Ridge_contour", "Lasso", "plots", "scaler".
