import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust

def design_matrix(x_flat, y_flat, poly):
    l = int((poly+1)*(poly+2)/2)                # Number of elements in beta
    X = np.ones((len(x_flat),l))

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x_flat**(i-k))*(y_flat**k)

    return X

class Regression:
    def __init__(self, x, y, z, maxpoly):
        self.xflat = x.ravel()
        self.yflat = y.ravel()
        self.zflat = z.ravel()

        self.X = design_matrix(x_flat, y_flat, maxpoly)
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.z, test_size = 0.2)


    def n_features(self, poly):
        return int( (poly + 1)*(poly + 2)/2 ) # + 1 ? "Nei" - Sara

    def scale(self, str_arg):
        """
        Possible args: Standard, MinMax, Mean, Robust
        """
        if lower(str_arg) == "none":
            self.scaled = False
            return None
        elif str_arg in ["Standard", "MinMax", "Mean", "Robust"]:
            self.scaled = True
            if str_arg in ["Standard", "MinMax", "Mean"]:
                scalefunc = eval( str_arg + "Scaler")
            if str_arg == "Normalizer":
                scalefunc = Normalizer

            self.X_train_scaled, self.X_test_scaled, self.z_train_scaled, self.z_test_scaled =\
                    scalefunc(self.X_train, self.X_test, self.z_train, self.z_test)
        else:
            print("invalid argument in scale() function.\
                    Possible args: Standard, MinMax, Mean, Normalizer")
            quit(1)


    def beta(self, lamb):
        I = np.eye( len( X_train[0, :]) )
        if self.scaled == False:
            beta = np.linalg.pinv(self.X_train.T @ self.X_train + lamb*I) @\
                    self.X_train.T @ self.z_train

        elif self.scaled == True:
            beta = np.linalg.pinv(self.X_train_scaled.T @ self.X_train_scaled + lamb*I) @\
                    self.X_train_scaled.T @ self.z_train_scaled

        return beta

    def OLS_Ridge(self, scale_arg = "none", scale = True, lamb):
        scale(scale_arg)

        beta = beta(self, lamb)
        z_model = self.X_train_scaled @ beta
        z_predict = self.X_test_scaled @ beta



    def Ridge(self):
        pass

    def GD(self, n_iterations, eta): # Gradient descent
        theta = np.random.randn(features, 1)
        #eta = 0.0025
        eta_GD = (1 / np.max(eig_vals))

        n_iterations = 100000

        z_train = self.z_train.reshape(-1, 1)

        #Define the cost function to be able to take the gradient using autograd
        def cost_func(X_train, theta, z_train):
            return (1/len(X_train[:, 0]))*((X_train @ theta)-z_train)**2      #np.sum(((X_train @ theta)-z_train)**2)



        for iter in range(n_iterations):

            gradients = self.X_train.T @ ((self.X_train @ theta)-self.z_train) + lamb * theta   #replace this with autograd
            #Should this be n_dpoints???   Removed at the start:  2.0/ n_dpoints *
            #derivative = egrad(cost_func, 1)
            #gradients = derivative(X_train, theta, z_train)   #take the derivative w.r.t. theta

            if abs(sum(gradients)) <= 0.00001:
                break
            theta -= eta_GD*gradients

        #print("GD theta: ", theta)

        z_predict = self.X_test @ theta
        z_model = self.X_train @ theta

        error_estimate(self, self.z_train, self.z_test, z_model, z_predict)



    def SGD(self, epochs, batch_size, eta): # Stochastic Gradient descent
        epochs = 1000
        batch_size = 5
        batches = int(len(X_train[:, 0])/batch_size)
        eta = 0.0025   #need to add some algorithm to scale the learning rate
        theta = np.random.randn(features, 1)

        z_train = self.z_train.reshape(-1, 1)

        for e in range (epochs):
            for b in range (batches):
                indices = np.random.randint(0, high = len(self.X_train[:, 0]), size = batch_size)
                X_b = self.X_train[indices]
                z_b = self.z_train[indices]
                gradient = X_b.T @ ((X_b @ theta)-z_b) + lamb * theta

                if abs(sum(gradient)) <= 0.00001:
                    break

                theta -= eta*gradient


        #print("SGD theta: ", theta)

        z_predict = self.X_test @ theta
        z_model = self.X_train @ theta

        error_estimate(self, self.z_train, self.z_test, z_model, z_predict)



    def error_estimate(self, z_train, z_test, z_model, z_predict):
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
