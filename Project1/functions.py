from scalers import *

#np.random.seed(2018)


def ThreeD_plot(x, y, z, title):
    fig_predict = plt.figure()
    ax_predict = fig_predict.gca(projection='3d')

    surf_predict = ax_predict.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig_predict.colorbar(surf_predict, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def design_matrix(x_flat, y_flat, poly):
    l = int((poly+1)*(poly+2)/2)		# Number of elements in beta
    X = np.ones((len(x_flat),l))

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[:,q+k] = (x_flat**(i-k))*(y_flat**k)

    return X


def scaling(X_train, X_test, z_train, z_test, scaler):

    if scaler == "standard" :
         X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerStandard(X_train, X_test, z_train, z_test)

    elif scaler == "minmax" :
         X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerMinMax(X_train, X_test, z_train, z_test)

    elif scaler == "mean" :
         X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerMean(X_train, X_test, z_train, z_test)

    elif scaler == "robust" :
         X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scalerRobust(X_train, X_test, z_train, z_test)

    else:
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = X_train, X_test, z_train, z_test


    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled



def OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, plot):    #Gjør Ridge, hvis lambda != 0

    #scaling = eval(scaler)
    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test, scaler)

    I = np.eye(len(X_train_scaled[0,:]))

    beta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled + lamb*I) @ X_train_scaled.T @ z_train_scaled    #use the pseudoinverse for the singular matrix

    #Generate the model z
    z_model = X_train_scaled @ beta

    #generate the prediction z
    z_predict = X_test_scaled @ beta
    #print(len(z_test_scaled), len(z_predict))

    if plot == "plot_prediction":
            #Plot the Prediction

            x_axis = np.linspace(0, 1, 20)
            y_axis = np.linspace(0, 1, 20)
            x_axis, y_axis = np.meshgrid(x_axis, y_axis)

            x_axis_flat = np.ravel(x_axis)
            y_axis_flat = np.ravel(y_axis)

            X_new = design_matrix(x_axis_flat, y_axis_flat, poly)

            z_new = X_new @ beta   #gir 1d kolonne

            z_new_grid = z_new.reshape(20, 20)   #make into a grid for plotting

            #ThreeD_plot(x_axis, y_axis, z_new_grid, "Prediction")

    return z_train_scaled, z_test_scaled, z_predict, z_model




def Lasso(X_train, X_test, z_train, z_test, scaler, lamb, poly, plot = False):

    X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scaling(X_train, X_test, z_train, z_test, scaler)

    RegLasso = linear_model.Lasso(lamb, tol=4e-1, max_iter=1e6)   #tol=1e-1, max_iter=1e7

    RegLasso.fit(X_train_scaled, z_train_scaled)

    z_model = RegLasso.predict(X_train_scaled)
    z_predict = RegLasso.predict(X_test_scaled)

    return z_train_scaled, z_test_scaled, z_predict, z_model



def Bootstrap(x, y, z, scaler, poly, B_runs, reg_method, lamb, dependency):

    X = design_matrix(x, y, poly)

    X_train_tot, X_test_tot, z_train, z_test = train_test_split(X, z, test_size = 0.2)#, random_state=2018)

    if dependency == "bias_variance":

        MSE_train = []
        MSE_test = []
        Bias = []
        Variance = []

        for degree in range(0, poly+1):

            n = int((degree + 1)*(degree + 2)/2) + 1

            X_train = X_train_tot[:, :n]
            X_test = X_test_tot[:, :n]

            z_predictions = np.zeros((len(z_test), B_runs))   #matrix containing the values for different bootstrap runs

            MSE_train_boot = []

            for i in range(B_runs):

                X_train_boot, z_train_boot = resample(X_train, z_train)

                if reg_method == "OLS" or reg_method == "Ridge":

                    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train_boot, X_test, z_train_boot, z_test, scaler, lamb, degree, "none")

                    MSE_train_boot.append(mean_squared_error(z_train_scaled, z_model))

                elif reg_method == "Lasso":
                    z_train_scaled, z_test_scaled, z_predict, z_model = Lasso(X_train_boot, X_test, z_train_boot, z_test, scaler, lamb, degree)

                    MSE_train_boot.append(mean_squared_error(z_train_scaled, z_model))


                z_predictions[:, i] = z_predict.ravel()
                z_test_scaled = z_test_scaled.reshape((-1, 1))


            MSE_train.append(np.mean(MSE_train_boot))

            mse_test = np.mean( np.mean((z_test_scaled - z_predictions)**2, axis=1, keepdims=True) )
            bias = np.mean( (z_test_scaled - np.mean(z_predictions, axis=1, keepdims=True))**2 )
            variance = np.mean( np.var(z_predictions, axis=1, keepdims=True) )

            MSE_test.append(mse_test)
            Bias.append(bias)
            Variance.append(variance)


        return MSE_train, MSE_test, Bias, Variance


    elif dependency == "lambda":

        n_lambdas = 200
        lambdas = np.logspace(-10, 5, n_lambdas)   #list of values

        MSE = []
        LAMBDA = []

        for lamb in lambdas:

            z_predictions = np.zeros((len(z_test), B_runs))

            for i in range(B_runs):

                X_train_boot, z_train_boot = resample(X_train_tot, z_train)

                if reg_method == "Ridge":

                    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train_boot, X_test_tot, z_train_boot, z_test, scaler, lamb, poly, "none")

                elif reg_method == "Lasso":

                    z_train_scaled, z_test_scaled, z_predict, z_model = Lasso(X_train_boot, X_test_tot, z_train_boot, z_test, scaler, lamb, poly)

                z_predictions[:, i] = z_predict.ravel()


            z_test_scaled = z_test_scaled.reshape((-1, 1))

            mse = np.mean(np.mean((z_test_scaled - z_predictions)**2, axis=1, keepdims=True))
            MSE.append(mse)
            LAMBDA.append(lamb)

        return MSE, LAMBDA



def CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, lamb, rng, dependency=None):

    X = design_matrix(x_flat, y_flat, poly)
    mse_cv = np.zeros(poly+1)
    scaling = eval(scaler)

    deg = 0
    #Permute the data
    perm = rng.permutation(np.arange(0, X.shape[0]))
    X_ = X[perm, :]
    z_ = z_flat[perm]
    for i in range(poly+1):

        X_current_poly = X_[:, :int((deg + 1) * (deg + 2) / 2)]


        kfolds = KFold(n_splits=k_fold)
        scores_KFold = np.zeros(k_fold)

        #Permute the data
        #perm = rng.permutation(np.arange(0, X_current_poly.shape[0]))
        #X_current_poly = X_current_poly[perm, :]
        #z_ = z_flat[perm]

        k = 0
        for train_inds, test_inds in kfolds.split(X_current_poly):

            X_train = X_current_poly[train_inds, :]
            z_train = z_[train_inds]
            #z_train = z_flat[train_inds]
            X_test = X_current_poly[test_inds, :]
            z_test = z_[test_inds]
            #z_test = z_flat[test_inds]

            #Scale the data
            #X_train_sc, X_test_sc, z_train_sc, z_test_sc = scaling(X_train, X_test, z_train, z_test)

            #OLS av X_train og z_train
            #betas = np.linalg.pinv(X_train_sc.T @ X_train_sc + lamb * np.eye(X_train_sc.shape[1])) @ X_train_sc.T @ z_train_sc
            z_train_sc, z_test_sc, z_pred, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, i, plot=None)

            #z_tilde = X_test_sc @ betas
            #print(f"polydeg : {deg}, k iter : {k}")
            #print(f"CV predict shape : {z_tilde.shape}")
            #print(f"CV test shape : {z_test.shape}")
            

            scores_KFold[k] = mean_squared_error(z_test_sc, z_pred)
            k += 1

        mse_cv[i] = np.mean(scores_KFold)
        deg += 1

    return mse_cv



if __name__  == "__main__":
    main("test")
