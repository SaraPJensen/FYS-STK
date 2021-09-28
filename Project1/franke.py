from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.evaluate import bias_variance_decomp

np.random.seed(2018)

def MSE(data, prediction):
    n = len(prediction)
    return (1/n) * np.sum((data - prediction)**2)


def R2(data, prediction):
    n = len(prediction)
    return 1 - (np.sum((data - prediction)**2))/np.sum((data - (np.sum(data))/n)**2)

def variance(prediction):
    #variance = np.var(prediction)
    variance = np.var(prediction)
    #print("Variance", variance)
    return variance

def bias(data, prediction):
    #bias = np.mean(data - np.mean(prediction)) #, axis=1, keepdims=True))**2
    bias = np.mean((data - np.mean(prediction))**2)
    #print("Bias ", bias)
    return bias


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


#Mortens design matrix
def design_matrix(x, y, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x),l))

    for i in range(0, poly+1):
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

def plot2(x, y, z, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title(title)

    plt.show()


def OLS(x, y, z, poly):
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    X = design_matrix(x, y, poly)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    #Scale the design matrix, do this in a separate function
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #scale the response variable
    z_train_scaled = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_scaled = (z_test - np.mean(z_train))/np.std(z_train)

    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix

    print(beta)

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


def bias_variance(poly, runs):

    MSE_test = np.zeros([poly,runs])
    Bias = np.zeros([poly,runs])
    Variance = np.zeros([poly,runs])
    polynomial = []

    x = np.sort(np.random.uniform(0, 1, 20))
    y = np.sort(np.random.uniform(0, 1, 20))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.1*np.random.randn(20, 20)


    for i in range(1, poly+1):

        for j in range(runs):

            z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid = OLS(x, y, z, i)

            MSE_test[i-1][j] = MSE(z_test_scaled, z_predict)
            Bias[i-1][j] = bias(z_train_scaled, z_predict)
            Variance[i-1][j] = variance(z_predict)

        polynomial.append(i)

    for k in range(poly):
        MSE_test[k][0] = np.mean(MSE_test[k])
        Bias[k][0] = np.mean(Bias[k])
        Variance[k][0] = np.mean(Variance[k])

    return MSE_test[:,0], Bias[:,0], Variance[:,0], polynomial



def tradeoff(poly, runs):

    MSE_train = np.zeros([poly,runs])
    MSE_test = np.zeros([poly,runs])
    R2_train = np.zeros([poly,runs])
    R2_test = np.zeros([poly,runs])
    polynomial = []

    #Blir det riktigere å plassere datagenereringen her?
    x = np.sort(np.random.uniform(0, 1, 20))
    y = np.sort(np.random.uniform(0, 1, 20))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.1*np.random.randn(20, 20)


    for i in range(1, poly+1):


        for j in range(runs):
            '''
            x = np.sort(np.random.uniform(0, 1, 20))
            y = np.sort(np.random.uniform(0, 1, 20))
            x, y = np.meshgrid(x, y)
            z = FrankeFunction(x, y) + 0.1*np.random.randn(20, 20)
            '''
            #Nå bruker den nye test/training data hver runde. Dette er jo egentlig cross-validation...
            z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid = OLS(x, y, z, i)

            MSE_train[i-1][j] = (MSE(z_train_scaled, z_model))
            R2_train[i-1][j] = R2(z_train_scaled, z_model)

            MSE_test[i-1][j] = (MSE(z_test_scaled, z_predict))
            R2_test[i-1][j] = (R2(z_test_scaled, z_predict))

        polynomial.append(i)

    for k in range(poly):
        MSE_train[k][0] = np.mean(MSE_train[k])
        R2_train[k][0] = np.mean(R2_train[k])
        MSE_test[k][0] = np.mean(MSE_test[k])
        R2_test[k][0] = np.mean(R2_test[k])


    return MSE_train[:,0], MSE_test[:,0], R2_train[:,0], R2_test[:,0], polynomial



def Bootstrap(x, y, z, poly, B):
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)


    R2 = []
    MSE = []
    Bias = []
    Variance = []

    for degree in poly:

        X = design_matrix(x, y, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

        z_predictions = ([(len(z_test)), B])

        for i in range(B):
            X_train_boot, z_train_boot = resample(X_train, z_train)

            #Scale the design matrix
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train_boot)
            X_test_scaled = scaler.transform(X_test)

            #scale the response variable
            z_train_scaled = (z_train_boot - np.mean(z_train_boot))/np.std(z_train_boot)
            z_test_scaled = (z_test - np.mean(z_train_boot))/np.std(z_train_boot)

            beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix

            #generate the prediction z
            z_predict = X_test_scaled @ beta   #bruker denne dataen til å regne ut feil, R2, MSE

            z_predictions[:, i] = z_predict


        error = np.mean( np.mean((z_test - z_predict)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_predict, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_predict, axis=1, keepdims=True) )


    return




n = 25
# Generate data
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.01*np.random.randn(n, n)

polynomial = 5

#plot(x, y, z, "Function")

z_test_scaled, z_train_scaled, z_predict, z_model, x_axis, y_axis, z_new_grid = OLS(x, y, z, polynomial)


#plot(x_axis, y_axis, z_new_grid, "Prediction")

#MSE_train, MSE_test, R2_train, R2_test, poly = tradeoff(polynomial, 100)


#Error, Bias, Variance, poly_degree = bias_variance(polynomial, 1)

'''
print('')
r2 = R2(z_test_scaled, z_predict)
print("R2, test: ", r2)
print('')
r2 = R2(z_train_scaled, z_model)
print("R2, train: ", r2)
print('')

mse = MSE(z_test_scaled, z_predict)
print("MSE, test: ", mse)
print('')
mse = MSE(z_train_scaled, z_model)
print("MSE, train: ", mse)
print('')
'''

'''

plt.plot(poly, MSE_test, label="Testing data", color='blue')
plt.plot(poly, MSE_train, label="Training data", color='red')
plt.xlabel("Degrees of polynomial")
plt.ylabel("Mean Squared Errod")
plt.legend()
plt.show()


plt.plot(poly, R2_test, label="Testing data", color='blue')
plt.plot(poly, R2_train, label="Training data", color='red')
plt.xlabel("Degrees of polynomial")
plt.ylabel("R squared")
plt.legend()
plt.show()


plt.plot(poly_degree, Bias, label="Bias", color='teal')
plt.plot(poly_degree, Variance, label="Variance", color='peru')
plt.plot(poly_degree, Error, label="Error", color='firebrick')
plt.xlabel("Degrees of polynomial")
#plt.ylabel("")
plt.legend()
plt.show()
'''
