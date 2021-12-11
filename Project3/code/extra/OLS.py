from Franke_data import *



class Regression:
    def __init__(self, X_train, X_test, z_train, z_test, poly):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test
        self.poly = poly

        #self.features = len(self.X_train[0,:])
        #self.datapoints = len(self.X_train[:, 0])


    def beta(self, X_train, z_train):
        beta = np.linalg.pinv(X_train.T @ X_train ) @ X_train.T @ z_train

        return beta


    def Bootstrap(self):

        MSE_train = []
        MSE_test = []
        Bias = []
        Variance = []
        B_runs = 200

        filename = "OLS_bias_var" + str(np.random.randint(0, 1000000))

        file = open(f"data_bv/{filename}.csv", "w")
        file.write("Polynomial,MSE_train,MSE_test,Bias,Variance\n")
        file.close()


        for degree in range(0, self.poly+1):

            print("Poly degree: ", degree)
            print()

            n = int((degree+1)*(degree+2)/2)
            print("n: ", n)

            X_train = self.X_train[:, :n]
            X_test = self.X_test[:, :n]

            print(len(X_train[0,:]))

            z_predictions = np.zeros((len(self.z_test), B_runs))   #matrix containing the values for different bootstrap runs

            MSE_train_boot = []

            for i in range(B_runs):

                X_train_boot, z_train_boot = resample(self.X_train, self.z_train)

                beta = self.beta(X_train_boot, z_train_boot)

                z_model = X_train_boot @ beta
                z_predict = self.X_test @ beta

                MSE_train_boot.append(mean_squared_error(self.z_train, z_model))

                z_predictions[:, i] = z_predict.ravel()

            z_test = self.z_test.reshape((-1, 1))

            mse_train = np.mean(MSE_train_boot)
            mse_test = np.mean( np.mean((z_test - z_predictions)**2, axis=1, keepdims=True) )
            bias = np.mean( (z_test - np.mean(z_predictions, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(z_predictions, axis=1, keepdims=True) )

            MSE_train.append(mse_train)
            MSE_test.append(mse_test)
            Bias.append(bias)
            Variance.append(variance)

            file = open(f"data_bv/{filename}.csv", "a")
            file.write(f"{degree},{mse_train},{mse_test},{bias},{variance}\n")
            file.close()

        return MSE_train, MSE_test, Bias, Variance



def main():
    poly = 20
    n_points = 20
    noise = 0.2
    design = "poly"

    X_train, X_test, z_train, z_test = Franke_data(n_points, noise, design, poly)

    OLS = Regression(X_train, X_test, z_train, z_test, poly)

    OLS.Bootstrap()

main()
