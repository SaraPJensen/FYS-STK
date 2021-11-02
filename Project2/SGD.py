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

n_dpoints = 40

noise = 0.2

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





poly = 8

features = int((poly + 1) * (poly + 2) / 2)


X_train = X_train_tot[:, :features]

X_test = X_test_tot[:, :features]

#To find the optimal eta, find the eigenvalues of XTX
matrix = X_train.T @ X_train
#print(np.shape(matrix))
eig_vals, eig_vecs = np.linalg.eig(matrix)

#Optimal learning rate is less than 1/max(eigvals)
print("Maximum learning rate: ", 1/max(eig_vals))

lamb = 0


#-------------------------------
#Regular gradient Descent
#-------------------------------
#Necessary variables for SGD
max_iter = 50
theta = np.random.randn(features, 1)
#eta = 0.0025
eta_GD = (1 / np.max(eig_vals))

n_iterations = 100000

z_train = z_train.reshape(-1, 1)

#Define the cost function to be able to take the gradient using autograd
def cost_func(X_train, theta, z_train):
    return (1/len(X_train[:, 0]))*((X_train @ theta)-z_train)**2      #np.sum(((X_train @ theta)-z_train)**2)



for iter in range(n_iterations):

    gradients = X_train.T @ ((X_train @ theta)-z_train) + lamb * theta   #replace this with autograd
    #Should this be n_dpoints???   Removed at the start:  2.0/ n_dpoints *
    #derivative = egrad(cost_func, 1)
    #gradients = derivative(X_train, theta, z_train)   #take the derivative w.r.t. theta

    if abs(sum(gradients)) <= 0.00001:
        break
    theta -= eta_GD*gradients

#print("GD theta: ", theta)

z_predict = X_test @ theta
z_model = X_train @ theta



print("Gradient descent")
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



#-----------------------------
#Stochastic Gradient Descent
#-----------------------------

#Parameters
epochs = 100
batch_size = 5
batches = int(len(X_train[:, 0])/batch_size)
eta = 0.0025   #need to add some algorithm to scale the learning rate
theta = np.random.randn(features, 1)


for e in range (epochs):
    for b in range (batches):
        indices = np.random.randint(0, high = len(X_train[:, 0]), size = batch_size)
        X_b = X_train[indices]
        z_b = z_train[indices]
        gradient = X_b.T @ ((X_b @ theta)-z_b) + lamb * theta

        if abs(sum(gradient)) <= 0.00001:
            break

        theta -= eta*gradient


#print("SGD theta: ", theta)

z_predict = X_test @ theta
z_model = X_train @ theta


print("Stochastic gradient descent")

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


#----------------------
#Sklearn SGD    - gives worse results - very strange...
#----------------------
SGDreg = SGDRegressor(max_iter = n_iterations, penalty = None, eta0 = eta, shuffle = True)
SGDreg.fit(X_train, np.ravel(z_train))
z_model = SGDreg.predict(X_train)
z_predict = SGDreg.predict(X_test)

print("Sklearn SGD")

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
