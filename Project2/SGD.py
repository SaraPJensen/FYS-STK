from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from numpy import linalg

from autograd import elementwise_grad as egrad
from autograd import grad


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

max_poly = 10
X = design_matrix(x_flat, y_flat, max_poly)

X_train_tot, X_test_tot, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)





poly = 5

features = int((poly + 1) * (poly + 2) / 2)


X_train = X_train_tot[:, :features]
X_test = X_test_tot[:, :features]

#To find the optimal eta, find the eigenvalues of XTX
matrix = X_train.T @ X_train
#print(np.shape(matrix))
eig_vals, eig_vecs = np.linalg.eig(matrix)

#Our own Stochastic gradient descent without minibatches
#Necessary variables for SGD
max_iter = 50
penalty = None
theta = np.random.randn(features, 1)
#eta = 0.025
eta = 1 / np.max(eig_vals)
n_iterations = 100000

z_train = z_train.reshape(-1, 1)


for iter in range(n_iterations):
    gradients = 2.0/ n_dpoints * X_train.T @ ((X_train @ theta)-z_train)

    if abs(sum(gradients)) <= 0.00001:
        print(sum(gradients))
        print("Iterations: ", iter)
        break

    theta -= eta*gradients



z_predict = X_test @ theta
z_model = X_train @ theta



mse_train = mean_squared_error(z_train, z_model)
print(f"MSE, train: {mse_train:.5}")
print('')
mse_test = mean_squared_error(z_test, z_predict)
print(f"MSE, test: {mse_test:.5}")
print('')

r2_train = r2_score(z_train, z_model)
print(f"R2, train: {r2_train:.5}")
print('')
r2_test = r2_score(z_test, z_predict)
print(f"R2, test: {r2_test:.5}")
print('')






'''
n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2.0* xi.T @ ((xi @ theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
print("theta from own sdg")
print(theta)


#Sklearn's SGD
sgdreg = SGDRegressor(max_iter = max_iter, penalty = penalty, eta0 = eta)
sgdreg.fit(X, z_flat)
'''
