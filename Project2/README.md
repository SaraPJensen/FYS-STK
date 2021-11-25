# Project 2 - Classification and Regression using Stochastic Gradient Descent, Feed Forward Neural Networks and Logistic Regression

This repository contains our code for project 2 of FYK-STK fall 2021. This README deatils the code, including how to run it.

## Running code
Run designated files for each section of the project with the selected function calls commented in. 

## Code Overview


### [Franke_data.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/Franke_data.py)
- Generates the continuous data-set analysed in the project.

#### Functions:
- design_matrix(): Generates the design matrix for a given polynomial degree
- FrankeFunction(): The Franke function without noise
- Franke_data(): Generates the data for the Franke function for given level of noise and number of data-points, returns design matrix and target data after train-test split


### [franke_regclass.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/franke_regclass.py)
Code for different regression methods: OLS, Ridge, Gradient descent and Stochastic gradient descent, all created as the same type of objects in the FrankeRegression class.

#### Class: 
FrankeRegression:
  - Functions:
    - beta(): generate beta
    - OLS_Ridge(): OLS or Ridge regression using bootstrapping
    - GD(): Gradient descent
    - SGD(): Stochastic gradient descent, returns either plots over epochs, MSE and R2 scores over epochs or the MSE and R2 scores
    - error(): returns the error for a given prediction and model
    - print_error(): prints the error for a given prediction and model


### [ex_A.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/ex_A.py)
All the necessary code for doing exercise A in the project. Parameters must be changed manually within the functions and the desired functions calls (found at the bottom of the file) commented in.  

#### Functions:
- OLS(): OLS and Ridge regression with bootstrapping
- GD_test(): Regression with gradient descent
- SGD_test(): Regression with stochastic gradient descent
- learning_rate(): Plots MSE history over epochs for constant and adaptive learning rate
- momentum(): Plots MSE history over epochs with and without momentum
- gridsearch(): Gridsearch which plots MSE as function of learning rate and regularisation parameter to find the optimal combination
- batchloop(): Plots MSE as function of batchsize



### [neural_new.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/neural_new.py)
Code for neural network, containing a class for the network and a class for the layers. 

#### Functions:
- sigmoid(): Sigmoid activation function
- relu(): ReLU activation function
- leaky_relu(): Leaky ReLU activation function
- classify(): classifies as 0 or 1, used for classification problems

#### Classes: 
- NeuralNetwork
  - Creates a neural network with the architecture given by the parameters.  
  - Functions:
    - activation(): Activation functions  
    - activation_derivative(): Derivatives of activation functions
    - cost_derivative(): Derivatives fo cost functions
    - feed_forward(): feeds the inputs forward in the network
    - backpropagation(): backpropagation algorithm for tuning the weights and biases, calls on hidden_layer.update_parameters() for each layer
    - model_training(): trains the network for the given training data, calls on feed_forward() and backpropagation()
    - prediction(): generates a prediction for the given input, be be used after model training

- hidden_layer
  - One layer-object for each hidden layer in the network, as well as the input and output layers. Contains information about the number of nodes, hidden weights and biases, as well as input and output to and from the layer.
  - Functions:
    -  update_parameters(): updates the weights and biases according to the parameters given. 



### [ex_B.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/ex_B.py)
All the necessary code for doing exercises B and C in the project. Parameters must be changed manually within the functions and the desired functions calls (found at the bottom of the file) commented in. 

#### Functions:
- benchmark_split(): Calculates the "optimal" MSE and R2 scores by comparing the noisy data-set with the Franke function
- test_network(): Generates a single network and returns the errors and plot of learning history if wanted
- gridsearch(): Gridsearch to find optimal combination of learning rate and regularisation parameter
- nodes(): Plots MSE over epochs for different number of nodes and layers
- activation(): Plots MSE over epoch for different activation functions
- weights(): Plots MSE over epoch for different weight initialisation methods
- learning(): Plots MSE over epoch for different learning rates
- batchsize(): Plots MSE as function of batchsize
- SKlearn(): Generates model and prediction using Sci-Kit learn's MLPRegressor. 

### [nn_classific.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/nn_classific.py)
Produces plots used in part d.
Necessary datafiles are stored in the repository such that no additional calculations should be on the users part. If rerunning calculations is desirable, add an additional argument when running the program.
In order to change the activation function and weight initialization change the variables in the file. Possible arguments for activation function, af = {'sigmoid', 'relu', 'leaky_relu'}. Possible arguments for weight initialization, wi = {'xavier', 'he', 'none', 'nielsen'}.
Changing the number of epochs and the lambda array is also necessary.
Sigmoid xavier should have 100 epochs, the relu's he should have 50 epochs. It is apparent from the comments which lambda arrays correspond to which combination. No other combination has been saved.

### [logreg.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/logreg.py)
Produces plots used in part e.
Necessary datafiles are stored in the repository such that no additional calculations should be needed on the users part. If rerunning calculations is desirable, add an additional argument when running the program. No changes in the file should be necessary.

### [score_plots.py](https://github.com/SaraPJensen/FYS-STK/blob/main/Project2/score_plots.py)
Produces plots used in part d.
Necessary datafiles are stored in the repository such that no additional calculations should be needed on the users part. Follow the instructions if preducing results is desirable.

For just producing plots: add the argument 'sigmoid' in order to plot results tied to the sigmoid/logistic activation function, or add the argument 'relu' in order to plot results tied to the relu/leaky relu activation functon. Add an additional argument DIFFERENT from 'train' in order to just plot the data.

For producing plots and recalculating data: add 'train' as an additional argument after the 'sigmoid'/'relu' argument.
