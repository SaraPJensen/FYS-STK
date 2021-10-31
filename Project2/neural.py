import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg
from autograd import elementwise_grad


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

def relu(z):
    if z < 0:
        return 0
    else:
        return z


def softmax(z):
    pass



#-----------------------------------------------
#This is what we actually want to get working!!
#-----------------------------------------------

class NeuralNetwork:
    def __init__(self, X_train, z_train, X_test, z_test, n_hidden_nodes, n_hidden_layers, n_output_nodes, epochs, batch_size, eta, lamb, activation_func = "sigmoid", cost_func = "MSE"):

        self.X_train = X_train
        self.z_train = z_train
        self.X_test = X_test
        self.z_test = z_test

        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers

        self.n_output_nodes = n_output_nodes
        self.epochs = epochs
        self.batch_size = batch_size

        self.eta = eta   #add a function which updates this for changing learning rate?
        self.lamb = lamb

        self.activation_func = activation_func

        self.cost = cost_func

        self.layers = []  #list of layer-objects

        #Let X_train be the first layer in the NN
        #What shape should X be? Isn't each input meant to be a number, not a vector?
        X = hidden_layer(0, 0)   #doesn't need weights and biases

        X.z_out = X_train

        #The first layer must have different dimensions since the number of features in X is (likely) different from the number of nodes
        layer1 = hidden_layer(self.n_hidden_nodes, self.n_features)

        output_layer = hidden_layer(self.n_hidden_nodes, self.n_output_nodes)

        self.layers.append(X)
        self.layers.append(layer1)

        for i in range (2, self.n_hidden_layers):
            i = hidden_layer(self.n_hidden_nodes, self.n_hidden_nodes)   #Assuming all layers have the same number of nodes
            self.layers.append(i)   #a list of layer-objects




    def activation(self, z):
        if self.activation_func == "sigmoid" or self.activation_func == "Sigmoid":
            return sigmoid(z)

        elif self.activation_func == "RELU" or self.activation_func == "relu" or self.activation_func == "Relu":
            return relu(z)

        elif self.activation_func == "softmax" or self.activation_func == "Softmax":
            return softmax(z)


    def cost(self, a_out, z_train):

        if self.cost == "MSE":
            return (a_out - z_train)**2

        elif self.cost == "accuracy"
            pass


    def feed_forward(self):
        previous = self.layers[0]

        for layer in (self.layers[1:]):
            layer.a_in = previous.a_out
            #In z_hidden, each row represents the outputs for a given layer
            z_hidden = previous.z_out @ layer.hidden_weights + layer.hidden_bias
            a_hidden = activation(z_hidden)

            layer.a_out = a_hidden  #update the matrix of z_values, containing all the inputs for the next layer

            previous = layer


    def backpropagation(self):
        #for layer in self.layers[1::-1]:
        error_output = output_layer.a_out - self.z_train

        weights = output_layer.a_out.T @ error_output
        bias_gradient = np.sum(error_output)

        for layer in self.layers[-2:0:-1]:  #don't include the output layer or input layer

            error = error_output @ layer.hidden_weights.T * layer.a_out * (1 - layer.a_out)

            weights_gradient = (layer.a_in, error)   #should it be a_in or a_out here??
            bias_gradient = np.sum(error) #, axis=0)

            layer.update_parameters(weights_gradient, bias_gradient, self.eta)



class hidden_layer:   #let each layer be associated with the weights and biases that come before it
    def __init__(self, n_hidden_nodes, n_features):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_features = n_features

        #Initialise weights and biases
        #initialise the weights according to a normal distribution
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_nodes)
        #The weigts should be a matrix where each column is the weights for a given node
        #Only one weight matrix per layer

        #the bias is a vector, where each element is the bias for one node
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01   #initialise all biases to a small number

        self.a_in = [[]]
        self.a_out = [[]]  #this should just be a matrix of indefinite size, just want to initialise it...


        #Update the weights and biases
        def update_parameters(self, weights_gradient, bias_gradient, eta):
            self.hidden_weights -= eta*weights_gradient
            self.hidden_bias -= eta*bias_gradient






















#---------------------------------------
#Initial architecture of neural network - based on Morten's code
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
