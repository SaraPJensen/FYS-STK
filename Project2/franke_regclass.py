from Franke_data import *



class FrankeRegression:
    def __init__(self, X_train, X_test, z_train, z_test):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

        self.features = len(self.X_train[0,:])
        self.datapoints = len(self.X_train[:, 0])


    def beta(self, X_train, z_train, lamb):
        I = np.eye( len( X_train[0, :]) )
        beta = np.linalg.pinv(X_train.T @ X_train + lamb*I) @ X_train.T @ z_train

        return beta


    def OLS_Ridge(self, lamb, B_runs, error_print="yes"):

        MSE_train = []
        MSE_test = []
        R2_train = []
        R2_test = []

        for i in range(B_runs):
            X_train_boot, z_train_boot = resample(self.X_train, self.z_train)

            beta = self.beta(X_train_boot, z_train_boot, lamb)

            z_model = X_train_boot @ beta
            z_predict = self.X_test @ beta

            MSE_train.append(mean_squared_error(z_train_boot, z_model))
            MSE_test.append(mean_squared_error(self.z_test, z_predict))

            R2_train.append(r2_score(z_train_boot, z_model))
            R2_test.append(r2_score(self.z_test, z_predict))


        mse_train = np.mean(MSE_train)
        mse_test = np.mean(MSE_test)
        r2_train = np.mean(R2_train)
        r2_test = np.mean(R2_test)

        if error_print == "yes":

            print(f"MSE, train: {mse_train:.5}")
            print('')
            print(f"MSE, test: {mse_test:.5}")
            print('')
            print(f"R2, train: {r2_train:.5}")
            print('')
            print(f"R2, test: {r2_test:.5}")
            print('')

        return mse_train, mse_test, r2_train, r2_test



    def GD(self, epochs, eta, momentum = 0, lamb = 0, learning_schedule = "no"): # Gradient descent

        theta = np.random.randn(self.features, 1)
        z_train = self.z_train.reshape(-1, 1)
        dtheta = 0
        eta0 = eta

        for e in range(epochs):

            if learning_schedule == "yes":
                if eta0 * (1 - e / epochs) > 0:
                    eta = eta0 * (1 - e / epochs)

            gradient = self.X_train.T @ ((self.X_train @ theta) - z_train) + lamb * theta
            if abs(sum(gradient)) <= 0.00001:
                break

            dtheta = momentum * dtheta - eta*gradient
            theta += dtheta

        z_model = self.X_train @ theta
        z_predict = self.X_test @ theta

        return z_model, z_predict



    def SGD(self, epochs, batch_size, eta, momentum = 0, lamb = 0, learning_schedule = "no", plot = "no"): # Stochastic Gradient descent
        batches = int(len(self.X_train[:, 0])/batch_size)

        theta = np.random.randn(self.features, 1)
        z_train = self.z_train.reshape(-1, 1)
        dtheta = 0

        eta0 = eta

        if plot == "yes" or plot == "data":
            Epochs = []
            mse_train = []
            mse_test = []
            r2_train = []
            r2_test = []


        for e in range (epochs):

            if learning_schedule == "yes":
                if eta0 * (1 - e / epochs) > 0:
                    eta = eta0 * (1 - e / epochs)

            indices = np.random.permutation(self.datapoints)
            indices = np.array_split(indices, batches)

            for b in range (batches):
                index = np.random.randint(batches)

                X_b = self.X_train[indices[index],:]   #pick out what rows to use
                z_b = z_train[indices[index]]
                gradient = X_b.T @ ((X_b @ theta)-z_b) + lamb * theta

                if abs(sum(gradient)) <= 0.00001:
                    break

                dtheta = momentum * dtheta - eta*gradient
                theta += dtheta


            if plot == "yes" or plot == "data":

                z_predict = self.X_test @ theta
                z_model = self.X_train @ theta

                mse_train.append(mean_squared_error(self.z_train, z_model))
                mse_test.append(mean_squared_error(self.z_test, z_predict))

                r2_train.append(r2_score(self.z_train, z_model))
                r2_test.append(r2_score(self.z_test, z_predict))

                Epochs.append(e)


        if plot == "yes":
            plt.plot(Epochs, mse_train, label = "MSE train")
            plt.plot(Epochs, mse_test, label = "MSE test")
            plt.xlabel("Epochs", size = 12)
            plt.ylabel("MSE", size = 12)
            plt.title(f"MSE for SGD on Franke function with noise 0.05", size = 12)
            plt.legend()
            plt.show()

            plt.plot(Epochs, r2_train, label = r"R$^2$ train")
            plt.plot(Epochs, r2_test, label = r"R$^2$ test")
            plt.xlabel("Epochs", size = 12)
            plt.ylabel(r"R$^2$", size = 12)
            plt.title(r"R$^2$ score for SGD on Franke function with noise 0.05", size = 12)
            plt.legend()
            plt.show()

        if plot == "data":   #return the lists
            return Epochs, mse_train, mse_test, r2_train, r2_test


        z_predict = self.X_test @ theta
        z_model = self.X_train @ theta

        return z_model, z_predict


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
