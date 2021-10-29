import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def design_matrix(x_flat, y_flat, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x_flat),l))

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x_flat**(i-k))*(y_flat**k)

    return X


np.random.seed(2018)

n_dpoints = 20

noise = 0

x = np.arange(0,1,1/n_dpoints)
y = np.arange(0,1,1/n_dpoints)

x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)

max_poly = 5
X = design_matrix(x_flat, y_flat, max_poly)

X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

#---------------------
#Activation functions
#---------------------
def sigmoid(z):
    return 1/(1 + np.exp(-z))


def RELU():
    pass

def Softmax():
    pass


#---------------------------------------
#Initial architecture of neural network
#---------------------------------------
#Variables:
n_inputs = len(X_train[:, 0])
n_features = len(X_train[0, :])
n_hidden_nodes = 50    #one bias for each node
n_hidden_layers = 10
n_output_nodes = 10

epochs=10
batch_size=100
eta=0.01


#Weights and biases in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_nodes)  #initialise the weights according to a norman distribution
hidden_bias = np.zeros(n_hidden_nodes)   #initialise all biases to 0

#Weights and biases in the output layer
output_weights = np.random.randn(n_hidden_nodes, n_output_nodes)
output_bias = np.zeros(n_output_nodes)


#Feed forward for each hidden layer

def feed_forward_train(X_train):

    #INPUT LAYER
    #weighted and biased inputs
    z_hidden = np.matmul(X_train, hidden_weights) + hidden_bias

    #activation function
    a_hidden = sigmoid(z_hidden)


    #OUTPUT LAYER
    z_output = np.matmul(a_hidden, output_weights) + output_bias

    exp_term = np.exp(z_output)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    return probabilities

def backpropagation(X_train, z_train):
    a_hidden, probabilities = feed_forward_train(X_train)

    #error in output layer
    error_output = probabilities - z_train

    #error in hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_hidden * (1-a_hidden)

    #gradients for output layer to optimise parameters
    output_weights_gradient = np.matmul(a_hidden.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)

    #gradients for hidden layer to optimise parameters
    hidden_weights_gradient = np.matmul(X_train.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis = 0)

    '''
    output_weights -= eta * output_weights_gradient
    output_bias -= eta * output_bias_gradient
    hidden_weights -= eta * hidden_weights_gradient
    hidden_bias -= eta * hidden_bias_gradient
    '''

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


class NeuralNetwork:
    def __init__(self, X_train, z_train, n_hidden_nodes, n_hidden_layers, n_output_nodes, epochs, batch_size, eta, lamb):

        self.X_train = X_train
        self.z_train = z_train

        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.n_output_nodes = n_output_nodes
        self.epochs = epochs
        self.batch_size = batch_size

        self.eta = eta
        self.lamb = lamb


class hidden_layer:
    def __init__(self, n_hidden_nodes, n_features):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_features = n_features

        #Initialise weights and biases
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_nodes)  #initialise the weights according to a norman distribution
        self.hidden_bias = np.zeros(self.n_hidden_nodes)   #initialise all biases to 0


        #Update the weights and biases
        def update_params(self, weights_gradient, bias_gradient, eta):
            self.hidden_weights -= eta*weights_gradient
            self.hidden_bias -= eta*bias_gradient
