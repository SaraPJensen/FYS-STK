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
import seaborn as sns


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



#Add momentum to the SGD

def scalerStandard(X_train, X_test, z_train, z_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #scale the response variable
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_train))/np.std(z_train)

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled



#---------------------
#Activation functions
#---------------------
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.where(z > 0, z, 0)

def leaky_relu(z):
    return np.where(z > 0, z, 0.01*z)

def classify(z):
    return np.where(z > 0.5, 1, 0)


#-----------------------------------------------
#This is what we actually want to get working!!
#-----------------------------------------------

class NeuralNetwork:
    def __init__(self, X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func = "sigmoid", cost_func = "MSE", dataset = "function", weight_init_method = "he"):

        self.X_train = X_train
        self.z_train_full = z_train
        self.z_train = z_train   #Use this when training on the whole dataset. This variable is redefined for each round when using SGD
        self.X_test = X_test
        self.z_test = z_test

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
        self.init_method = weight_init_method

        self.layers = []  #list of layer-objects, containing all the hidden layers plus the input and output layers


        #------------------------------------------
        #Setting up the architecture of the network
        #------------------------------------------
        #Let X_train be the first layer in the NN
        self.input = hidden_layer(1, 1, self.init_method)   #doesn't need weights and biases
        self.input.a_out = self.X_train

        #The first layer must have different dimensions since the number of features in X is (likely) different from the number of nodes
        self.layer1 = hidden_layer(self.n_features, self.hidden_nodes[0], self.init_method)   #The input layer has two nodes, each containing a vector of all the datapoints
        self.output_layer = hidden_layer(self.hidden_nodes[-1], 1, self.init_method)


        self.layers.append(self.input)
        self.layers.append(self.layer1)


        for i in range(1, len(self.hidden_nodes)):  #Want (n_hidden_layers + 2) layers in total, start at 1, since the first layer is already made
            layer = hidden_layer(self.hidden_nodes[i-1], self.hidden_nodes[i], self.init_method)   #Assuming all layers have the same number of nodes
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
            layer.z_hidden = layer.a_in @ layer.hidden_weights + layer.hidden_bias

            output = self.activation(layer.z_hidden)

            layer.a_out = output  #update the matrix of z_values, containing all the inputs for the next layer

            previous = layer

        if self.dataset == "function":
            layer.a_out = layer.z_hidden  # no activation func for output layer when a function is fitted, only for classification

        elif self.dataset.lower() == "classification":
            layer.a_out = sigmoid(layer.z_hidden)    #Always use sigmoid in the last layer for classification



    def backpropagation(self):

        grad_cost = self.cost_derivative(self.output_layer.a_out)
        grad_activation = self.activation_derivative(self.output_layer.z_hidden)

        if self.dataset == "function":
            grad_activation = 1

        error_output = grad_cost * grad_activation

        self.output_layer.error = error_output
        next_layer = self.output_layer


        for layer in self.layers[-2:0:-1]:  #don't include the output layer or input layer
            #calculate all the errors before the weights and biases are updated
            error = (next_layer.error @ next_layer.hidden_weights.T) * self.activation_derivative(layer.z_hidden)

            layer.error = error
            next_layer = layer


        for layer in self.layers[-1:0:-1]:  #don't include the input layer, which has no weights and biases
            weights_gradient = np.matmul(layer.a_in.T, layer.error) + self.lamb * layer.hidden_weights    #is this correct?? Shouldn't this just be added in the cost function at the last layer?

            bias_gradient = np.sum(layer.error, axis=0)

            layer.update_parameters(weights_gradient, bias_gradient, self.eta)



    def model_training(self, method = "SGD", plot = "yes"):

        if method == "SGD" and self.dataset == "classification":
            if plot == "yes":
                Epochs = []
                train_accuracy = []
                test_accuracy = []

            for e in range(0, self.epochs):
                indices = np.random.permutation(self.n_datapoints)
                indices = np.array_split(indices, self.batches)

                for b in range(self.batches):
                    index = np.random.randint(self.batches)

                    self.input.a_out = self.X_train[indices[index],:]   #pick out what rows to use
                    self.z_train = self.z_train_full[indices[index]]

                    self.feed_forward()
                    self.backpropagation()

                if plot == "yes":

                    z_model = self.prediction(self.X_train)
                    z_predict = self.prediction(self.X_test)


                    model_sum = np.sum(z_model)   #it starts returning nan pretty quickly, which is weird
                    if np.isnan(model_sum):
                        break


                    z_classified = classify(z_model)
                    results = np.column_stack((self.z_train_full, z_classified))
                    accuracy = np.abs(self.z_train_full.ravel() - z_classified.ravel())
                    total_wrong = sum(accuracy)
                    percentage = (len(accuracy) - total_wrong)/len(accuracy)

                    z_predict_class = classify(z_predict)
                    results_test = np.column_stack((self.z_test, z_predict_class))
                    accuracy_test = np.abs(self.z_test.ravel() - z_predict_class.ravel())
                    total_wrong_test = sum(accuracy_test)
                    percentage_test = (len(accuracy_test) - total_wrong_test)/len(accuracy_test)


                    Epochs.append(e)
                    train_accuracy.append(percentage)
                    test_accuracy.append(percentage_test)


            if plot == "yes":
                plt.plot(Epochs, train_accuracy, label = "Accuracy train")
                plt.plot(Epochs, test_accuracy, label = "Accuracy test")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title(f"Accuracy using {self.activation_func} as activation function")
                plt.legend()
                plt.show()

                return(train_accuracy[-1], test_accuracy[-1])


            return(train_accuracy[-1], test_accuracy[-1]) # Error when plot='no'


        elif method == "SGD" and self.dataset == "function":

            if plot == "yes":
                Epochs = []
                mse_train = []
                mse_test = []
                r2_train = []
                r2_test = []

            for e in range(1, self.epochs):
                indices = np.random.permutation(self.n_datapoints)
                indices = np.array_split(indices, self.batches)

                #maybe add a test for convergence of errors?


                for b in range(self.batches):
                    index = np.random.randint(self.batches)

                    self.input.a_out = self.X_train[indices[index],:]   #pick out what rows to use
                    self.z_train = self.z_train_full[indices[index]]

                    self.feed_forward()
                    self.backpropagation()

                if plot == "yes":

                    z_model = self.prediction(self.X_train)
                    z_predict = self.prediction(self.X_test)

                    mse_train.append(mean_squared_error(self.z_train_full, z_model))
                    mse_test.append(mean_squared_error(self.z_test, z_predict))

                    r2_train.append(r2_score(self.z_train_full, z_model))
                    r2_test.append(r2_score(self.z_test, z_predict))

                    Epochs.append(e)


            if plot == "yes":
                plt.plot(Epochs, mse_train, label = "MSE train")
                plt.plot(Epochs, mse_test, label = "MSE test")
                plt.xlabel("Epochs")
                plt.ylabel("MSE")
                plt.title(f"MSE using {self.activation_func} as activation function")
                plt.legend()
                plt.show()


                plt.plot(Epochs, r2_train, label = "R2 train")
                plt.plot(Epochs, r2_test, label = "R2 test")
                plt.xlabel("Epochs")
                plt.ylabel("R2")
                plt.title(f"R2 score using {self.activation_func} as activation function")
                plt.legend()
                plt.show()


            z_model = self.prediction(self.X_train)
            z_predict = self.prediction(self.X_test)

            mse_train = (mean_squared_error(self.z_train_full, z_model))
            mse_test = (mean_squared_error(self.z_test, z_predict))

            r2_train = (r2_score(self.z_train_full, z_model))
            r2_test = (r2_score(self.z_test, z_predict))


            return mse_train, mse_test, r2_train, r2_test



        if method == "GD":
            for i in range (self.epochs):

                self.feed_forward()
                self.backpropagation()




    def prediction(self, data):
        self.input.a_out = data   #The input layer is replaced with the test data
        self.feed_forward()

        return self.output_layer.a_out  #z_prediction



    def grid_search(self, eta, lamb):

        if self.dataset == "function":

            self.eta = eta
            self.lamb = lamb










class hidden_layer:   #let each layer be associated with the weights and biases that come before it
    def __init__(self, n_previous_nodes, n_hidden_nodes, init_method):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_previous_nodes = n_previous_nodes
        self.init_method = init_method

        #Initialise weights and biases
        #The weights can be initialised according to different functions

        np.random.seed(123)

        if self.init_method.lower() == "he":
            self.hidden_weights = np.random.normal(scale=(np.sqrt(2)/np.sqrt(self.n_previous_nodes)), size=(self.n_previous_nodes, self.n_hidden_nodes))

        elif self.init_method.lower() == "xavier":
            bound = np.sqrt(6)/(np.sqrt(self.n_previous_nodes))

            self.hidden_weights = np.random.uniform(-bound, bound, size = (self.n_previous_nodes, self.n_hidden_nodes))

        elif self.init_method.lower() == "none":
            self.hidden_weights = np.random.randn(self.n_previous_nodes, self.n_hidden_nodes)

        elif self.init_method.lower() == "homemade":
            self.hidden_weights = np.random.randn(self.n_previous_nodes, self.n_hidden_nodes) / (np.sqrt(2)*np.sqrt(self.n_hidden_nodes))    #this is the wrong expression for he, but it works bloody well....

            #self.hidden_weights = np.random.randn(self.n_previous_nodes, self.n_hidden_nodes) / self.n_hidden_nodes


        #the bias is a vector, where each element is the bias for one node
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01   #initialise all biases to a small number

        self.a_in = None
        self.a_out = None  #this should just be a matrix of indefinite size, just want to initialise it...
        self.z_hidden = None

        self.error = None


    #Update the weights and biases
    def update_parameters(self, weights_gradient, bias_gradient, eta):
        self.hidden_weights -= eta*weights_gradient   #This has been done wrong!!!
        self.hidden_bias -= eta*bias_gradient





def main(data):
    #np.random.seed(1234)
    print('')
    print('')
    print('')
    print("Restart")

    if data.lower() == "franke":
        #------------------------------
        #Franke function analysis
        #------------------------------
        #np.random.seed(123)

        n_dpoints = 30
        noise = 0.2

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
        eta = 0.07    #0.05 and 0.01 works well for leaky relu and relu, 0.03 works for sigmoid
        batch_size = 30
        epochs = 100
        lamb = 0

        activation_func = "relu"
        cost_func = "mse"
        dataset = "function"
        training_method = "SGD"
        weight_init_method = "he"



        '''
        #np.random.seed(123)
        Neural = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
        Neural.model_training(training_method)


        z_model = Neural.prediction(X_train)
        z_predict = Neural.prediction(X_test)

        results = np.column_stack((z_train, z_model))
        #print(results)

        print("Train MSE: ", mean_squared_error(z_train, z_model))
        print("Train R2 score: ", r2_score(z_train, z_model))
        print('')
        print("Test MSE: ", mean_squared_error(z_test, z_predict))
        print("Test R2 score: ", r2_score(z_test, z_predict))
        '''



        #gridsearch for lambda and eta

        eta_min = 1e-7
        eta_max = 1
        eta_n = 5
        eta = np.logspace(-7, 0, eta_n)
        #eta = np.linspace(eta_min, eta_max, eta_n)

        lamb_min = 1e-7
        lamb_max = 10
        lamb_n = 5

        #lamb = np.linspace(lamb_min, lamb_max, lamb_n)
        lamb = np.logspace(-7, 0, lamb_n)

        mse_results = np.zeros((len(lamb), len(eta)))   #each row corresponds to one value of lambda, each column to a value of eta
        r2_results = np.zeros((len(lamb), len(eta)))


        for e in range(len(eta)):
            for l in range(len(lamb)):
                np.random.seed(123)
                NN = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta[e], lamb[l], activation_func, cost_func, dataset, weight_init_method)
                mse_train, mse_test, r2_train, r2_test = NN.model_training(training_method, "no")

                mse_results[l, e] = mse_test  #row l, column e
                r2_results[l, e] = r2_test

                print(e, l)


        min = np.min(mse_results)
        index = np.where(mse_results == min)
        print("Min MSE: ", min)
        print("Min eta: ", eta[index[1]])
        print("Min lambda: ", lamb[index[0]])

        results = [eta, lamb, mse_results]

        print(lamb)
        print(eta)

        eta = np.round(np.log10(eta), 3)
        lamb = np.round(np.log10(lamb), 3)


        ax = sns.heatmap(mse_results, xticklabels = eta, yticklabels = lamb,  annot=True, cmap="YlGnBu")
        #ax.set_xticks(eta)
        #ax.set_yticks(lamb)

        ax.set_xlabel(r"log10$\eta$")
        ax.set_ylabel(r"log10$\lambda$")

        plt.show()













    elif data.lower() == "cancer":
        #-------------------------------------
        #Breast cancer analysis
        #-------------------------------------
        #np.random.seed(1234)
        cancer = load_breast_cancer(return_X_y=False, as_frame=False)
        X = cancer.data   #cancer.data is the design matrix, dimensions 569x30
        z = cancer.target  #cancer.target is the target values, 1=Malign, 0=Benign


        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        X_train, X_test, z_train_useless, z_test_useless = scalerStandard(X_train, X_test, z_train, z_test)


        hidden_nodes = [30, 30]   #This is a list of the number of nodes in each hidden layer
        eta = 0.00001   #0.00001 or 0.0001 works well for sigmoid, 0.01 for relu and leaky relu   (0.0001 for relu and xavier)
        batch_size = 80
        epochs = 1000
        lamb = 0

        activation_func = "relu"
        cost_func = "accuracy"
        dataset = "classification"
        weight_init_method = "he"     #use homemade for relu and leaky_relu


        Neural = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
        percentage_train, percentage_test = Neural.model_training("SGD")

        print("Train accuracy: ", percentage_train)

        print("Test accuracy: ", percentage_test)




if __name__ == "__main__":
    main("cancer")
