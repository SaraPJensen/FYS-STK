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

'''
def ThreeD_plot(x, y, z, title):
    fig_predict = plt.figure()
    ax_predict = fig_predict.gca(projection='3d')

    surf_predict = ax_predict.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig_predict.colorbar(surf_predict, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
'''



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

    if scaler == "Standard" :
        something

    elif scaler == "MinMax" :
        something

    elif scaler == "Normalise" :
        something

    elif scaler == "Robust" :
        something

    else:
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = X_train, X_test, z_train, z_test

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled



def OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb):    #Gjør Ridge, hvis lambda != 0

    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test)

    I = np.eye(len(X_train_scaled[0,:]))

    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled - lamb*I) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix

    #Generate the model z
    z_model = X_train_scaled @ beta

    #generate the prediction z
    z_predict = X_test_scaled @ beta

    return z_train_scaled, z_test_scaled, z_predict, z_model



def Lasso(X_train, X_test, z_train, z_test, scaler, lamb):

    #Legg til if-statement for ulike skaleringer
    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test)


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


def main(n, polynomial):
    # Generate data
    polynomial = 20
    n = 20
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.5*np.random.randn(n, n)


    return 0
