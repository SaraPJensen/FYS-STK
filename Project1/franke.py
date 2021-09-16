from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def MSE(data, prediction):
    n = len(prediction)
    return (1/n) * np.sum((data - prediction)**2)

def R2(data, prediction):
    n = len(prediction)
    return 1 - (np.sum((data - prediction)**2))/np.sum((data - (np.sum(data))/n)**2)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

'''
def design_matrix(x, y):  #Makes the design matrix for a fifth order polynomial, dimension 20x21

    #Stacks column vectors together. axis=-1 means they are columns and not rows
    X = np.stack((np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, (x**2)*y, (y**2)*x, x**4, y**4, (x**3)*y, (y**3)*x, (x**2)*(y**2), x**5, y**5, (x**4)*y, (y**4)*x, (x**3)*(y**2), (y**3)*(x**2)), axis=-1)
    return X
'''


#Mortens design matrix
def design_matrix(x, y, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x),l))

    for i in range(1,poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X



def plot(x, y, z, title):
    fig_predict = plt.figure()
    ax_predict = fig_predict.gca(projection='3d')
    # Plot the surface.
    surf_predict = ax_predict.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig_predict.colorbar(surf_predict, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()






def OLS(x, y, z, poly):
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    X = design_matrix(x, y, poly)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    #Scale the design matrix
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #scale the response variable
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_train))/np.std(z_train)

    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix


    #Generate the model z
    z_model = X_train_scaled @ beta

    #generate the prediction z
    z_predict = X_test_scaled @ beta   #bruker denne dataen til å regne ut feil, R2, MSE


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

    return z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid





def tradeoff(poly, runs):

    MSE_train = []
    MSE_test = []
    R2_train = []
    R2_test = []
    polynomial = []


'''
    for i in range(1, poly+1):


        for j in range(runs):

            x = np.sort(np.random.uniform(0, 1, 20))
            y = np.sort(np.random.uniform(0, 1, 20))
            x, y = np.meshgrid(x, y)
            z = FrankeFunction(x, y) + 0.1*np.random.randn(20, 20)

            z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid = OLS(x, y, z, i)


        MSE_train.append(MSE(z_model, z_train_scaled))
        R2_train.append(R2(z_model, z_train_scaled))

        MSE_test.append(MSE(z_predict, z_test_scaled))
        R2_test.append(R2(z_predict, z_test_scaled))



    polynomial.append(i)


    return MSE_train, MSE_test, R2_train, R2_test, polynomial
'''


# Generate data
x = np.sort(np.random.uniform(0, 1, 20))
y = np.sort(np.random.uniform(0, 1, 20))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.1*np.random.randn(20, 20)

polynomial = 5

z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid = OLS(x, y, z, polynomial)


plot(x_axis, y_axis, z_new_grid, "Prediction")

'''
print('')
r2 = R2(z_predict, z_test_scaled)
print("R2, test: ", r2)
print('')
r2 = R2(z_model, z_train_scaled)
print("R2, train: ", r2)
print('')

mse = MSE(z_predict, z_test_scaled)
print("MSE, test: ", mse)
print('')
mse = MSE(z_model, z_train_scaled)
print("MSE, train: ", mse)
print('  ')




MSE_train, MSE_test, R2_train, R2_test, polynomial = tradeoff(x, y, z, 30)


plt.plot(polynomial, MSE_test, label="Testing data", color='blue')
plt.plot(polynomial, MSE_train, label="Training data", color='red')
plt.xlabel("Degrees of polynomial")
plt.ylabel("Mean Squared Errod")
plt.legend()
plt.show()


plt.plot(polynomial, R2_test, label="Testing data", color='blue')
plt.plot(polynomial, R2_train, label="Training data", color='red')
plt.xlabel("Degrees of polynomial")
plt.ylabel("R squared")
plt.legend()
plt.show()
'''
