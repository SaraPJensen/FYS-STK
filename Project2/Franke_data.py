import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from numpy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample


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


def Franke_data(n_dpoints, noise, design, poly = 0):
    np.random.seed(1234567)   #this was used for high noise

    x = np.arange(0,1,1/n_dpoints)
    y = np.arange(0,1,1/n_dpoints)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + noise*np.random.randn(n_dpoints, n_dpoints)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    if design.lower() == "poly":
        X = design_matrix(x_flat, y_flat, poly)

    elif design.lower() == "stack":
        X = np.column_stack((x_flat, y_flat))

    np.random.seed(2018)   #this was used for high noise
    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.2)

    return X_train, X_test, z_train, z_test
