import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from numpy import linalg
import seaborn as sns
import matplotlib.pyplot as plt


def design_matrix(x_flat, y_flat, poly):
    l = int((poly+1)*(poly+2)/2)                #Number of elements in beta, features in X
    X = np.ones((len(x_flat),l))

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x_flat**(i-k))*(y_flat**k)
    return X


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def Franke_data(n_dpoints, noise, poly):

    np.random.seed(2018)
    x = np.arange(0,1,1/n_dpoints)
    y = np.arange(0,1,1/n_dpoints)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    X = design_matrix(x_flat, y_flat, poly)

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.2)

    return X_train, X_test, z_train, z_test


class FrankeRegression:
    def __init__(self, X_train, X_test, z_train, z_test):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

        self.features = len(self.X_train[0,:])
        self.datapoints = len(self.X_train[:, 0])


    def beta(self, lamb):
        I = np.eye( len( self.X_train[0, :]) )
        beta = np.linalg.pinv(self.X_train.T @ self.X_train + lamb*I) @ self.X_train.T @ self.z_train

        return beta


    def OLS_Ridge(self, lamb):
        beta = self.beta(lamb)

        z_model = self.X_train @ beta
        z_predict = self.X_test @ beta

        return z_model, z_predict


    def max_eta(self):
        matrix = self.X_train.T @ self.X_train
        eig_vals, eig_vecs = np.linalg.eig(matrix)
        eta = (1 / np.max(eig_vals))  #calculate the maximum eta

        return eta


    def GD(self, epochs, eta = 0, momentum = 0, lamb = 0): # Gradient descent

        theta = np.random.randn(self.features, 1)
        if eta == 0:
            eta = self.max_eta()

        z_train = self.z_train.reshape(-1, 1)
        dtheta = 0

        for iter in range(epochs):
            gradient = self.X_train.T @ ((self.X_train @ theta) - z_train) + lamb * theta
            if abs(sum(gradient)) <= 0.00001:
                break

            dtheta = momentum * dtheta - eta*gradient
            theta += dtheta

        z_model = self.X_train @ theta
        z_predict = self.X_test @ theta

        return z_model, z_predict



    def SGD(self, epochs, batch_size, eta = 0, momentum = 0, lamb = 0): # Stochastic Gradient descent
        batches = int(len(self.X_train[:, 0])/batch_size)

        if eta == 0:
            eta = self.max_eta()

        theta = np.random.randn(self.features, 1)
        z_train = self.z_train.reshape(-1, 1)
        dtheta = 0


        for e in range (epochs):
            indices = np.random.permutation(self.datapoints)
            indices = np.array_split(indices, batches)

            for b in range (batches):
                index = np.random.randint(batches)

                X_b = self.X_train[indices[index],:]   #pick out what rows to use
                z_b = z_train[indices[index]]

                gradient = X_b.T @ ((X_b @ theta)-z_b) + lamb * theta

                if abs(sum(gradient)) <= 0.00001:
                    break

                dtheta = momentum * dtheta - eta*gradient
                theta += dtheta


        z_predict = self.X_test @ theta
        z_model = self.X_train @ theta

        return z_model, z_predict


    def error(self, z_model, z_predict):
        mse_train = mean_squared_error(self.z_train, z_model)
        mse_test = mean_squared_error(self.z_test, z_predict)
        r2_train = r2_score(self.z_train, z_model)
        r2_test = r2_score(self.z_test, z_predict)

        return mse_train, mse_test, r2_train, r2_test




    def print_error(self, z_model, z_predict):
        mse_train = mean_squared_error(self.z_train, z_model)
        print(f"MSE, train: {mse_train:.5}")
        print('')
        mse_test = mean_squared_error(self.z_test, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')

        r2_train = r2_score(self.z_train, z_model)
        print(f"R2, train: {r2_train:.5}")
        print('')
        r2_test = r2_score(self.z_test, z_predict)
        print(f"R2, test: {r2_test:.5}")
        print('')
