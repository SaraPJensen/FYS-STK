import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer# scalerStandard, scalerMinMax, scalerMean, scalerRobust
from sklearn.metrics import mean_squared_error as MSE, r2_score as R2

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

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
        self.xf = x.ravel()
        self.yf = y.ravel()
        self.zf = z.ravel()

        self.X = design_matrix(self.xf, self.yf, maxpoly)
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.zf, test_size = 0.2)

        self.maxpoly = maxpoly
        self.predicts = {}
        self.models = {}


    def n_features(self, poly):
        return int( (poly + 1)*(poly + 2)/2 ) # + 1 ? "Nei" - Sara

    def scale(self, str_arg):
        """
        Possible args: Standard, MinMax, Mean, Robust, none
        """
        if str_arg.lower() == "none":
            self.scaled = False
            self.X_train_scaled, self.X_test_scaled, self.z_train_scaled, self.z_test_scaled =\
                    self.X_train, self.X_test, self.z_train, self.z_test
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


    def OLSR_beta(self, lamb):
        I = np.eye( len( self.X_train[0, :]) )

        if self.scaled == False:
            beta = np.linalg.pinv(self.X_train.T @ self.X_train + lamb*I) @\
                    self.X_train.T @ self.z_train

        elif self.scaled == True:
            beta = np.linalg.pinv(self.X_train_scaled.T @ self.X_train_scaled + lamb*I) @\
                    self.X_train_scaled.T @ self.z_train_scaled

        return beta

    def OLS_Ridge(self, scale_arg = "none", scale = True, lamb = 0):
        self.scale(scale_arg)

        beta = self.OLSR_beta(lamb)
        #z_model = self.X_train_scaled @ beta
        #z_predict = self.X_test_scaled @ beta

        self.OLSRmodel = self.X_train_scaled @ beta
        self.OLSRpred = self.X_test_scaled @ beta

        if lamb == 0:
            self.predicts['OLS'] = self.OLSRpred
            self.models['OLS'] = self.OLSRmodel
        elif lamb > 0:
            self.predicts['Ridge'] = self.OLSRpred
            self.models['Ridge'] = self.OLSRmodel

    def eta(self):
        #Add some optionality for changing the learning rate
        pass

    def GD(self, n_iterations = 100000, eta = 0.0025, lamb = 0): # Gradient descent
        features = self.n_features(self.maxpoly)
        theta = np.random.randn(features, 1)
        #eta = 0.0025
        eig_vals, eig_vecs = np.linalg.eig(self.X_train.T @ self.X_train)
        eta_GD = (1 / np.max(eig_vals))

        #n_iterations = 100000

        z_train = self.z_train.reshape(-1, 1)

        #Define the cost function to be able to take the gradient using autograd
        def cost_func(X_train, theta, z_train):
            return (1/len(X_train[:, 0]))*((X_train @ theta)-z_train)**2      #np.sum(((X_train @ theta)-z_train)**2)



        for iter in range(n_iterations):

            grad = self.X_train.T @ ((self.X_train @ theta)-z_train) + lamb * theta   #replace this with autograd
            #Should this be n_dpoints???   Removed at the start:  2.0/ n_dpoints *
            #derivative = egrad(cost_func, 1)
            #gradients = derivative(X_train, theta, z_train)   #take the derivative w.r.t. theta

            if np.linalg.norm(grad) <= 0.00001:
                break

            theta -= eta_GD * grad

        #print("GD theta: ", theta)

        #z_predict = self.X_test @ theta
        #z_model = self.X_train @ theta

        #error_estimate(self, self.z_train, self.z_test, z_model, z_predict)
        self.GDpred = self.X_test @ theta
        self.GDmodel = self.X_train @ theta

        #self.predicts.append(self.GDpred)
        #self.models.append(self.GDmodel)
        self.predicts['GD'] = self.GDpred
        self.models['GD'] = self.GDmodel


    def SGD(self, epochs = 1000, batch_size = 5, eta = 0.0025, lamb = 0): # Stochastic Gradient descent
        #epochs = 1000
        features = self.n_features(self.maxpoly)
        #batch_size = 5
        batches = int(len(self.X_train[:, 0])/batch_size)
        #eta = 0.0025   #need to add some algorithm to scale the learning rate
        theta = np.random.randn(features, 1)

        z_train = self.z_train.reshape(-1, 1)

        for e in range (epochs):
            indices = np.random.permutation(len(self.X_train[:, 0]))
            indices = np.array_split(indices, batches)
            for b in range (batches):
                #indices = np.random.randint(0, high = len(self.X_train[:, 0]), size = batch_size)
                X_b = self.X_train[indices[b]]
                z_b = z_train[indices[b]]
                grad = X_b.T @ ((X_b @ theta)-z_b) + lamb * theta

                if np.linalg.norm(grad) <= 0.00001:
                    break

                theta -= eta*grad


        #print("SGD theta: ", theta)

        self.SGDpred = self.X_test @ theta
        self.SGDmodel = self.X_train @ theta
    
        if lamb == 0:
            self.predicts['SGD'] = self.SGDpred
            self.models['SGD'] = self.SGDmodel
        elif lamb > 0:
            self.predicts['SGD_lamb'] = self.SGDpred
            self.models['SGD_lamb'] = self.SGDmodel


    def get_error(self):
        for i in self.predicts:
            print(i)
            print(f"MSE, pred: {MSE(self.predicts[i], self.z_test)}")
            print(f"MSE, model: {MSE(self.models[i], self.z_train)}")
            print(f"R2, pred: {R2(self.predicts[i], self.z_test)}")
            print(f"R2, model: {R2(self.models[i], self.z_train)}\n\n")
            


        #print(MSE(self.OLSRpred, self.z_test))
        #print(MSE(self.GDpred, self.z_test))

    '''
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
    '''

if __name__ == "__main__":
    n = 20
    x = np.random.rand(n)
    y = np.random.rand(n)

    x, y = np.meshgrid(x, y)

    noise = 0.15
    z = FrankeFunction(x, y) + noise * np.random.randn(n, n)

    maxpoly = 10
    reg = Regression(x, y, z, maxpoly)

    #reg.OLS_Ridge()
    #reg.OLS_Ridge(lamb = 0.001)
    #reg.GD()
    epochs = 1000
    batch_size = 5
    reg.SGD(epochs = epochs, batch_size = batch_size, lamb = 0.01)

    #print(f"epochs = {epochs}, batch size = {batch_size}")
    reg.SGD(epochs = epochs, batch_size = batch_size)
    reg.get_error()
