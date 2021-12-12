from Franke_data import *



class FrankeRegression:
    def __init__(self, X_train, X_test, z_train, z_test, poly):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test
        self.poly = poly

        self.features = len(self.X_train[0,:])
        self.datapoints = len(self.X_train[:, 0])


    def beta(self, X_train, z_train):
        beta = np.linalg.pinv(X_train.T @ X_train ) @ X_train.T @ z_train

        return beta



    def Bootstrap(self):

        MSE_train = []
        MSE_test = []
        Bias = []
        Variance = []
        B_runs = 100

        for degree in range(0, poly+1):

            print("Poly degree: ", degree)
            print()

            n = int((poly+1)*(poly+2)/2)

            X_train = self.X_train[:, :n]
            X_test = self.X_test[:, :n]

            z_predictions = np.zeros((len(self.z_test), B_runs))   #matrix containing the values for different bootstrap runs

            MSE_train_boot = []

            for i in range(B_runs):

                X_train_boot, z_train_boot = resample(self.X_train, self._train)

                beta = self.beta(X_train_boot, z_train_boot)

                z_model = X_train_boot @ beta
                z_predict = self.X_test @ beta

                MSE_train_boot.append(mean_squared_error(z_train_scaled, z_model))

                z_predictions[:, i] = z_predict.ravel()
                z_test_scaled = z_test_scaled.reshape((-1, 1))


            filename = "OLS_bias_" + str(np.random.randint(0, 1000000))

            file = open(f"data_bv/{filename}.csv", "w")
            file.write("MSE_train,MSE_test,Bias,Variance \n")
            file.close()

            mse_train = np.mean(MSE_train_boot)
            mse_test = np.mean( np.mean((z_test_scaled - z_predictions)**2, axis=1, keepdims=True) )
            bias = np.mean( (z_test_scaled - np.mean(z_predictions, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(z_predictions, axis=1, keepdims=True) )

            MSE_train.append(mse_train)
            MSE_test.append(mse_test)
            Bias.append(bias)
            Variance.append(variance)

            file = open(f"data_bv/{filename}.csv", "a")
            file.write(f"{mse_train},{mse_test},{bias},{variance}\n")
            file.close()

        return MSE_train, MSE_test, Bias, Variance





    def error(self, z_model, z_predict):
        mse_train = mean_squared_error(self.z_train, z_model)
        mse_test = mean_squared_error(self.z_test, z_predict)
        r2_train = r2_score(self.z_train, z_model)
        r2_test = r2_score(self.z_test, z_predict)

        return mse_train, mse_test, r2_train, r2_test



    def print_error(self, z_model, z_predict):
        mse_train = mean_squared_error(self.z_train, z_model)
        print(f"MSE, train: {mse_train:.5}")
        print('')
        mse_test = mean_squared_error(self.z_test, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')

        r2_train = r2_score(self.z_train, z_model)
        print(f"R2, train: {r2_train:.5}")
        print('')
        r2_test = r2_score(self.z_test, z_predict)
        print(f"R2, test: {r2_test:.5}")
        print('')
