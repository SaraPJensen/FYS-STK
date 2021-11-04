import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust
from math import exp, sqrt
from random import random, seed
import matplotlib.pyplot as plt
from numpy import linalg
from autograd import elementwise_grad
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.datasets import load_breast_cancer


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
    #print("input: ", z)
    #print("RElU returns :", np.where(z > 0, z, 0))
    return np.where(z > 0, z, 0)

def leaky_relu(z):
    return np.where(z > 0, z, 0.01*z)

def classify(z):
    return np.where(z > 0.5, 1, 0)


#-----------------------------------------------
#This is what we actually want to get working!!
#-----------------------------------------------

class NeuralNetwork:
    def __init__(self, X_train, z_train, hidden_nodes, epochs, batch_size, eta, lamb, activation_func = "sigmoid", cost_func = "MSE", dataset = "function"):

        self.X_train = X_train
        self.z_train_full = z_train
        self.z_train = z_train   #Use this when training on the whole dataset. This variable is redefined for each round when using SGD


        self.hidden_nodes = hidden_nodes

        self.n_features = len(self.X_train[0,:])
        self.n_datapoints = len(self.X_train[:, 0])

        self.epochs = epochs
        self.batch_size = batch_size
        self.batches = int(len(self.X_train[:, 0])/self.batch_size)



        self.eta = eta   #add a function which updates this for changing learning rate?
        self.lamb = lamb

        self.activation_func = activation_func
        self.cost_func = cost_func
        self.dataset = dataset

        self.layers = []  #list of layer-objects, containing all the hidden layers plus the input and output layers


        #------------------------------------------
        #Setting up the architecture of the network
        #------------------------------------------
        #Let X_train be the first layer in the NN
        self.input = hidden_layer(0, 0)   #doesn't need weights and biases
        self.input.a_out = self.X_train

        #The first layer must have different dimensions since the number of features in X is (likely) different from the number of nodes
        self.layer1 = hidden_layer(self.n_features, self.hidden_nodes[0])   #The input layer has two nodes, each containing a vector of all the datapoints
        self.output_layer = hidden_layer(self.hidden_nodes[-1], 1)


        self.layers.append(self.input)
        self.layers.append(self.layer1)


        for i in range(1, len(self.hidden_nodes)):  #Want (n_hidden_layers + 2) layers in total, start at 1, since the first layer is already made
            layer = hidden_layer(self.hidden_nodes[i-1], self.hidden_nodes[i])   #Assuming all layers have the same number of nodes
            self.layers.append(layer)   #list of layer-objects


        self.layers.append(self.output_layer)




    def activation(self, z):
        if self.activation_func.lower() == "sigmoid":
            return sigmoid(z)

        elif self.activation_func.lower() == "relu":
            return relu(z)

        elif self.activation_func.lower() == "leaky_relu":
            return leaky_relu(z)



    def cost(self, z_model): #Do we need this?
        if self.cost.lower() == "mse":
            return (z_model - self.z_train)**2

        elif self.cost.lower() == "accuracy":
            return - self.z_train * np.log(z_model) + (1 - self.z_train) * np.log(1 - z_model)




    def activation_derivative(self, z):
        if self.activation_func.lower() == "sigmoid":
            return sigmoid(z)*(1-sigmoid(z))

        elif self.activation_func.lower() == "relu":
            return np.where(z > 0, 1, 0)

        elif self.activation_func.lower() == "leaky_relu":
            return np.where(z > 0, 1, 0.01)




    def cost_derivative(self, z_model):
        if self.cost_func.lower() == "mse":
            return (-2/self.n_datapoints)*(self.z_train.reshape(z_model.shape) - z_model)

        elif self.cost_func.lower() == "accuracy":
            return (z_model - self.z_train.reshape(z_model.shape))/(z_model*(1-z_model))




    def feed_forward(self):
        previous = self.layers[0]

        for layer in (self.layers[1:]):
            layer.a_in = previous.a_out

            #In z_hidden, each column represents a node, each row a datapoint
            layer.z_hidden = previous.a_out @ layer.hidden_weights + layer.hidden_bias
            a_hidden = self.activation(layer.z_hidden)

            layer.a_out = a_hidden  #update the matrix of z_values, containing all the inputs for the next layer

            previous = layer

        if self.dataset == "function":
            layer.a_out = layer.z_hidden  # no activation func for output layer when a function is fitted, only for classification


    def backpropagation(self):

        if self.dataset == "function":
            error_output = self.cost_derivative(self.output_layer.a_out)  #No activation function used in the last layer

        elif self.dataset == "classification":
            grad_cost = self.cost_derivative(self.output_layer.a_out)
            grad_activation = self.activation_derivative(self.output_layer.z_hidden)
            error_output = grad_cost*grad_activation

        self.output_layer.error = error_output
        next_layer = self.output_layer


        for layer in self.layers[-2:0:-1]:  #don't include the output layer or input layer
            #calculate all the errors before the weights and biases are updated
            error = (next_layer.error @ next_layer.hidden_weights.T) * self.activation_derivative(layer.z_hidden)

            layer.error = error
            next_layer = layer


        for layer in self.layers[-1:0:-1]:  #don't include the input layer, which has no weights and biases
            weights_gradient = np.matmul(layer.a_in.T, layer.error)

            bias_gradient = np.sum(layer.error, axis=0)


            if self.lamb > 0.0:
                weights_gradient += self.lamb * layer.hidden_weights   #is this correct??

            layer.update_parameters(weights_gradient, bias_gradient, self.eta)



    def model_training(self, method = "SGD"):

        if method == "SGD":
            #indices = np.arange(self.n_datapoints)

            for e in range(self.epochs):
                #shuffle X_train
                #Split X_train i batcher k_folds split/array split

                for b in range(self.batches):
                    #plukk minibatch uten replacement
                    #Minibatching uten replacement er muligens bedre

                    indices = np.random.randint(0, high = self.n_datapoints-1, size = batch_size)
                    #current_datapoints = np.random.choice(indices, size=self.batch_size, replace=False)

                    self.input.a_out = self.X_train[indices,:]   #pick out what rows to use
                    self.z_train = self.z_train_full[indices]

                    #print("X_train shape :", self.input.a_out.shape)

                    self.feed_forward()
                    self.backpropagation()

            return self.output_layer.a_out #z_model


        if method == "GD":
            for i in range (self.epochs):

                self.feed_forward()
                self.backpropagation()

                #if np.abs(self.weights_gradient) < 0.01:
                #    break

        return self.output_layer.a_out #z_model




    def prediction(self, test_data):
        self.input.a_out = test_data   #The input layer is replaced with the test data
        self.feed_forward()

        return self.output_layer.a_out  #z_prediction





class hidden_layer:   #let each layer be associated with the weights and biases that come before it
    def __init__(self, n_previous_nodes, n_hidden_nodes):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_previous_nodes = n_previous_nodes

        #Initialise weights and biases
        #Initialise the weights according to a normal distribution - should probably scale the values with
        self.hidden_weights = np.random.randn(self.n_previous_nodes, self.n_hidden_nodes)#/n_hidden_nodes

        #the bias is a vector, where each element is the bias for one node
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01   #initialise all biases to a small number

        self.a_in = None
        self.a_out = None  #this should just be a matrix of indefinite size, just want to initialise it...
        self.z_hidden = None

        self.error = None

    #Update the weights and biases
    def update_parameters(self, weights_gradient, bias_gradient, eta):
        self.hidden_weights -= eta*weights_gradient
        self.hidden_bias -= eta*bias_gradient





def main(data):
    np.random.seed(1234)
    print('')
    print('')
    print('')
    print("Restart")

    if data.lower() == "franke":
        #------------------------------
        #Franke function analysis
        #------------------------------
        n_dpoints = 20
        noise = 0.02

        x = np.arange(0,1,1/n_dpoints)
        y = np.arange(0,1,1/n_dpoints)

        x, y = np.meshgrid(x, y)

        z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

        x_flat = np.ravel(x)
        y_flat = np.ravel(y)
        z_flat = np.ravel(z)

        X = np.column_stack((x_flat, y_flat))


        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)


        hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
        eta = 0.001
        batch_size = 32
        #iterations = 1000
        epochs = 5000
        lamb = 0

        activation_func = "sigmoid"
        cost_func = "mse"
        dataset = "function"
        training_method = "GD"

        np.random.seed(1234)
        Neural = NeuralNetwork(X_train, z_train, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset)
        z_model = Neural.model_training(training_method)
        z_predict = Neural.prediction(X_test)

        results = np.column_stack((z_train, z_model))
        #print(results)

        print("Train MSE: ", mean_squared_error(z_model, z_train))
        print("Train R2 score: ", r2_score(z_model, z_train))
        print('')
        print("Test MSE: ", mean_squared_error(z_predict, z_test))
        print("Test R2 score: ", r2_score(z_predict, z_test))



        Epochs = []
        mse_train = []
        mse_test = []
        r2_train = []
        r2_test = []

        for i in range(1000, 20000, 1000):
            np.random.seed(1234)
            epochs = i
            Neural = NeuralNetwork(X_train, z_train, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset)
            z_model = Neural.model_training(training_method)
            z_predict = Neural.prediction(X_test)

            mse_train.append(mean_squared_error(z_model, z_train))
            mse_test.append(mean_squared_error(z_predict, z_test))

            r2_train.append(r2_score(z_model, z_train))
            r2_test.append(r2_score(z_predict, z_test))
            print("Epochs: ", epochs)
            print('')
            print("Train MSE: ", mean_squared_error(z_model, z_train))
            print("Train R2 score: ", r2_score(z_model, z_train))
            print('')
            print("Test MSE: ", mean_squared_error(z_predict, z_test))
            print("Test R2 score: ", r2_score(z_predict, z_test))
            print('')

            Epochs.append(i)



        plt.plot(Epochs, mse_train, label = "MSE train")
        plt.plot(Epochs, mse_test, label = "MSE test")
        #plt.scatter(x, y, label="Testing data", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title(f"MSE using {activation_func} as activation function")
        plt.legend()
        plt.show()


        plt.plot(Epochs, r2_train, label = "R2 train")
        plt.plot(Epochs, r2_test, label = "R2 test")
        #plt.scatter(x, y, label="Testing data", color='b')
        plt.xlabel("Epochs")
        plt.ylabel("R2")
        plt.title(f"R2 score using {activation_func} as activation function")
        plt.legend()
        plt.show()





    elif data.lower() == "cancer":
        #-------------------------------------
        #Breast cancer analysis
        #-------------------------------------

        np.random.seed(1234)
        cancer = load_breast_cancer(return_X_y=False, as_frame=False)

        #cancer.data is the design matrix, dimensions 569x30
        #cancer.target is the target values, 1=Malign, 0=Benign

        X = cancer.data
        z = cancer.target


        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)


        hidden_nodes = [50, 50, 50, 50]   #This is a list of the number of nodes in each hidden layer
        eta = 0.00001
        batch_size = 32
        #iterations = 1000
        epochs = 500
        lamb = 0

        activation_func = "sigmoid"
        cost_func = "accuracy"
        dataset = "classification"


        Neural = NeuralNetwork(X_train, z_train, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset)
        z_model = Neural.model_training("GD")

        z_classified = classify(z_model)
        results = np.column_stack((z_train, z_classified))
        accuracy = np.abs(z_train.ravel() - z_classified.ravel())
        total_wrong = sum(accuracy)
        percentage = (len(accuracy) - total_wrong)/len(accuracy)

        #print("Training results")
        #print(results)
        print('')

        z_predict = Neural.prediction(X_test)
        z_predict_class = classify(z_predict)
        results_test = np.column_stack((z_test, z_predict_class))
        accuracy_test = np.abs(z_test.ravel() - z_predict_class.ravel())
        total_wrong_test = sum(accuracy_test)
        percentage_test = (len(accuracy_test) - total_wrong_test)/len(accuracy_test)


        #print("Test results")
        #print(results_test)

        print("Train accuracy: ", percentage)

        print("Test accuracy: ", percentage_test)



main("cancer")
