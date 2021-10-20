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

    def OLS(self, scale_arg = "none", scale = True):
        scale(scale_arg)


        pass

    def ridge(self):
        pass

    def GD(self): # Gradient descent
        pass

    def SGD(self): # Stochastic Gradient descent
        pass
