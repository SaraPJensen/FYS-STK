from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scalers import *

np.random.seed(2018)


def ThreeD_plot(x, y, z, title):
    fig_predict = plt.figure()
    ax_predict = fig_predict.gca(projection='3d')

    surf_predict = ax_predict.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig_predict.colorbar(surf_predict, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()




def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



def design_matrix(x, y, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x),l))

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X


def scaling(X_train, X_test, z_train, z_test, scaler):

    if scaler == "standard" :
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerStandard(X_train, X_test, z_train, z_test)

    elif scaler == "minmax" :
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerMinMax(X_train, X_test, z_train, z_test)

    else:
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = X_train, X_test, z_train, z_test


    '''
    elif scaler == "normalise" :
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaler...(X_train, X_test, z_train, z_test)

    elif scaler == "robust" :
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaler...(X_train, X_test, z_train, z_test)
    '''


    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled



def OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, plot):    #Gjør Ridge, hvis lambda != 0

    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test, scaler)

    I = np.eye(len(X_train_scaled[0,:]))

    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled - lamb*I) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix

    #Generate the model z
    z_model = X_train_scaled @ beta

    #generate the prediction z
    z_predict = X_test_scaled @ beta

    if plot == "plot_prediction":
            #Plot the Prediction
            scaler = StandardScaler()
            scaler.fit(X_train)

            #Plotter, genererer nye punkter for x, y, z
            x_axis = np.linspace(0, 1, 20)
            y_axis = np.linspace(0, 1, 20)
            x_axis, y_axis = np.meshgrid(x_axis, y_axis)

            x_axis_flat = np.ravel(x_axis)
            y_axis_flat = np.ravel(y_axis)

            X_new = design_matrix(x_axis_flat, y_axis_flat, poly)
            X_new_scaled = scaler.transform(X_new)

            z_new = X_new_scaled @ beta   #gir 1d kolonne

            z_new_scaled = (z_new - np.mean(z_train))/np.std(z_train)  #scale

            z_new_grid = z_new_scaled.reshape(20, 20)   #make into a grid for plotting

            ThreeD_plot(x_axis, y_axis, z_new_grid, "Prediction")

    return z_train_scaled, z_test_scaled, z_predict, z_model



def Lasso(X_train, X_test, z_train, z_test, scaler, lamb):

    #Legg til if-statement for ulike skaleringer
    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test, scaler)


    RegLasso = linear_model.Lasso(lamb, fit_intercept=False)

    RegLasso.fit(X_train_scaled, z_train_scaled)

    z_predict = RegLasso.predict(X_test_scaled)

    return z_train_scaled, z_test_scaled, z_predict, z_model




def Bootstrap(x, y, z, scaler, poly, B_runs, metode, lamb):
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    MSE = []
    Bias = []
    Variance = []

    for degree in poly:

        X = design_matrix(x, y, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
        z_predictions = ([(len(z_test)), B])

        for i in range(B):
            X_train_boot, z_train_boot = resample(X_train, z_train)

            #Legg til if-statements for ulike modeller
            z_train_scaled, z_test_scaled, z_predict, z_model = model(X_train, X_test, z_train, z_test, scaler, lamb)

            z_predictions[:, i] = z_predict

        error = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_predict, axis=1, keepdims=True) )

    return MSE, Bias, Variance


def CrossVal(x, y, z, scaler, poly, k_fold, metode, lamb):

    return mse, bias, variance


def main():
    # Generate data
    n = 25
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.01*np.random.randn(n, n)

    poly = 5

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    X = design_matrix(x_flat, y_flat, poly)

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

    '''
    Exercise 1
    '''
    '''
    #Plot the graph
    ThreeD_plot(x, y, z, "Function")

    #Plot prediction
    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, "standard", 0, poly, "plot_prediction")
    '''

    






main()
