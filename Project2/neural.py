import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg
from autograd import elementwise_grad
from sklearn.metrics import mean_squared_error, r2_score


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



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
    def __init__(self, X_train, z_train, X_test, z_test, n_hidden_nodes, n_hidden_layers, epochs, batch_size, eta, lamb, activation_func = "sigmoid", cost_func = "MSE"):

        self.X_train = X_train
        self.z_train = z_train
        self.X_test = X_test
        self.z_test = z_test

        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers  #number of hidden layers
        self.n_features = len(self.X_train[0,:])
        self.n_datapoints = len(self.X_train[:, 0])

        #self.n_output_nodes = n_output_nodes   #Don't need this
        self.epochs = epochs
        self.batch_size = batch_size

        self.eta = eta   #add a function which updates this for changing learning rate?
        self.lamb = lamb

        self.activation_func = activation_func
        self.cost_func = cost_func

        self.layers = []  #list of layer-objects, containing all the hidden layers plus the input and output layers



        #------------------------------------------
        #Setting up the architecture of the network
        #------------------------------------------

        #Let X_train be the first layer in the NN
        #What shape should X be? Isn't each input meant to be a number, not a vector?
        self.input = hidden_layer(0, 0)   #doesn't need weights and biases

        self.input.a_out = self.X_train


        #The first layer must have different dimensions since the number of features in X is (likely) different from the number of nodes
        self.layer1 = hidden_layer(self.n_features, self.n_hidden_nodes)

        self.output_layer = hidden_layer(self.n_hidden_nodes, 1)

        #self.output_layer = hidden_layer(self.n_hidden_nodes, self.n_hidden_nodes)

        self.layers.append(self.input)
        self.layers.append(self.layer1)


        for i in range (1, self.n_hidden_layers):  #Want (n_hidden_layers + 2) layers in total
            #Make a list of the number of nodes in each hidden layer
            #
            i = hidden_layer(self.n_hidden_nodes, self.n_hidden_nodes)   #Assuming all layers have the same number of nodes
            self.layers.append(i)   #list of layer-objects


        self.layers.append(self.output_layer)

        #print("Output hidden bias: ", self.output_layer.hidden_bias)



    def activation(self, z):
        if self.activation_func == "sigmoid" or self.activation_func == "Sigmoid":
            return sigmoid(z)

        elif self.activation_func == "RELU" or self.activation_func == "relu" or self.activation_func == "Relu":
            return relu(z)

        elif self.activation_func == "softmax" or self.activation_func == "Softmax":
            return softmax(z)


    def cost(self, a_out, z_train): #Do we need this?
        if self.cost == "MSE":
            return (a_out - z_train)**2

        elif self.cost == "accuracy":
            pass



    def activation_derivative(self, z):
        if self.activation_func == "sigmoid" or self.activation_func == "Sigmoid":
            return sigmoid(z)*(1-sigmoid(z))

        elif self.activation_func == "RELU" or self.activation_func == "relu" or self.activation_func == "Relu":
            if z < 0:
                return 0
            else:
                return 1

        elif self.activation_func == "softmax" or self.activation_func == "Softmax":
            pass



    def cost_derivative(self, z_model):
        if self.cost_func == "MSE":

            return (-2/self.n_datapoints)*(self.z_train.reshape(z_model.shape) - z_model)

        elif self.cost_func == "accuracy":
            pass



    def feed_forward(self):
        previous = self.layers[0]


        for layer in (self.layers[1:]):
            layer.a_in = previous.a_out

            #In z_hidden, each row represents the outputs for a given layer
            z_hidden = previous.a_out @ layer.hidden_weights + layer.hidden_bias
            a_hidden = self.activation(z_hidden)


            layer.a_out = a_hidden  #update the matrix of z_values, containing all the inputs for the next layer

            previous = layer

        layer.a_out = z_hidden  # no activation func for output layer


    def backpropagation(self):
        #for layer in self.layers[1::-1]:
        #error_output = self.output_layer.a_out - self.z_train.reshape(self.output_layer.a_out.shape)   #Wrong expression

        grad_cost = self.cost_derivative(self.output_layer.a_out)

        #grad_cost = (-2/self.n_datapoints)*(self.z_train.reshape(self.output_layer.a_out.shape) - self.output_layer.a_out)

        grad_activation = self.activation_derivative(self.output_layer.a_out)
        #grad_activation = sigmoid(self.output_layer.a_out)*(1-sigmoid(self.output_layer.a_out))
        #make this dependent on the cost function, this is the derivative of the cost function, which also includes the derivative of the cost function


        error_output = grad_cost# * grad_activation



        weights_gradient = self.output_layer.a_in.T @ error_output
        bias_gradient = np.sum(error_output)


        next_layer = self.output_layer


        self.output_layer.update_parameters(weights_gradient, bias_gradient, self.eta)


        error_output = grad_cost * grad_activation
        next_error = error_output

        for layer in self.layers[-2:0:-1]:  #don't include the output layer or input layer


            error = (next_error @ next_layer.hidden_weights.T) * self.activation_derivative(layer.a_out)
            #layer.a_out * (1 - layer.a_out)   #The last term is the derivative of the activation funciton

            layer.error = error

            next_error = error

            next_layer = layer



        for layer in self.layers[-2:0:-1]:  #don't include the output layer or input layer

            weights_gradient = np.matmul(layer.a_in.T, layer.error)
            bias_gradient = np.sum(layer.error, axis=0)

            if self.lamb > 0.0:
                self.weights_gradient += self.lamb * self.weights_gradient

            layer.update_parameters(weights_gradient, bias_gradient, self.eta)





    def model_training(self, method = "SGD"):

        if method == "SGD":
            #Use stochastic gradient descent to train the model
            pass

        if method == "test":
            for i in range (100):

                self.feed_forward()
                self.backpropagation()

                #if np.abs(self.weights_gradient) < 0.01:
                #    break

        return self.output_layer.a_out #z_model



    def prediction(self):
        self.X.z_out = self.X_test   #The input layer is replaced with the test data
        self.feed_forward()

        return self.output_layer.a_out  #z_prediction







class hidden_layer:   #let each layer be associated with the weights and biases that come before it
    def __init__(self, n_features, n_hidden_nodes):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_features = n_features

        #Initialise weights and biases
        #Initialise the weights according to a normal distribution
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_nodes)
        #The weight should be a matrix where each column is the weights for a given node
        #Only one weight matrix per layer

        #the bias is a vector, where each element is the bias for one node
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01   #initialise all biases to a small number

        self.a_in = None
        self.a_out = None  #this should just be a matrix of indefinite size, just want to initialise it...

        self.error = None

    #Update the weights and biases
    def update_parameters(self, weights_gradient, bias_gradient, eta):
        self.hidden_weights -= eta*weights_gradient
        self.hidden_bias -= eta*bias_gradient





np.random.seed(2018)

n_dpoints = 5

noise = 0

x = np.arange(0,1,1/n_dpoints)
y = np.arange(0,1,1/n_dpoints)

x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)


X = np.column_stack((x_flat, y_flat))

X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)



n_hidden_nodes = 3
n_hidden_layers = 1
eta = 0.03
batch_size = 50
epochs = 10
lamb = 0

print('')
print('')
print('')
print('')
print('')

Neural = NeuralNetwork(X_train, z_train, X_test, z_test, n_hidden_nodes, n_hidden_layers, epochs, batch_size, eta, lamb, activation_func = "sigmoid", cost_func = "MSE")
z_model = Neural.model_training("test")

z_model = z_model.ravel()



results = np.column_stack((z_train, z_model))

print(results)


print("MSE: ", mean_squared_error(z_model, z_train))
print("R2 score: ", r2_score(z_model, z_train))
