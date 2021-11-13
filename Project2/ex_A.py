from franke_regclass import *

X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0, poly=7, design = "poly")


def OLS():
    print()
    X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0.5, poly=3)
    print("OLS error:")
    OLS = FrankeRegression(X_train, X_test, z_train, z_test)
    OLS.OLS_Ridge(lamb = 0, B_runs = 1000, error_print = "yes")


    print()
    X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0.5, poly=7)
    print("Ridge error:")
    Ridge = FrankeRegression(X_train, X_test, z_train, z_test)
    Ridge.OLS_Ridge(lamb = 0.15, B_runs = 1000, error_print = "yes")


def GD_test():

    GD = FrankeRegression(X_train, X_test, z_train, z_test)
    z_model, z_predict = GD.GD(1000, eta = 0, momentum = 0, lamb = 0)
    print("Gradient descent error:")
    GD.print_error(z_model, z_predict)

    print()

def SGD_test():
    SGD = FrankeRegression(X_train, X_test, z_train, z_test)
    z_model, z_predict = SGD.SGD(epochs = 1000, eta = 0.00481487, batch_size = 35, momentum = 0, lamb = 0.00127427, plot = "yes")
    #z_model, z_predict = SGD.SGD(epochs = 500, eta = 0.01656155, batch_size = 26, momentum = 0, lamb = 0.0078476, plot = "no")

    print("Stochastic gradient descent error:")
    SGD.print_error(z_model, z_predict)


def learning_rate():
    constant_high = FrankeRegression(X_train, X_test, z_train, z_test)
    Epochs_h, mse_train_h, mse_test_h, r2_train_h, r2_test_h = constant_high.SGD(epochs = 1000, batch_size = 30, eta = 0.00481487, momentum = 0, lamb = 0.00127427, learning_schedule = "no", plot = "data")

    #constant_low = FrankeRegression(X_train, X_test, z_train, z_test)
    #Epochs_l, mse_train_l, mse_test_l, r2_train_l, r2_test_l = constant_low.SGD(epochs = 1000, batch_size = 30, eta = 0.01656155*0.5, momentum = 0, lamb = 0.0078476, learning_schedule = "no", plot = "data")

    schedule = FrankeRegression(X_train, X_test, z_train, z_test)
    Epochs_s, mse_train_s, mse_test_s, r2_train_s, r2_test_s = schedule.SGD(epochs = 1000, batch_size = 30, eta = 0.00481487, momentum = 0, lamb = 0.00127427, learning_schedule = "yes", plot = "data")

    plt.plot(Epochs_h[3:], mse_test_h[3:], label = r"$\eta$ = 0.004815")
    #plt.plot(Epochs_l[2:], mse_test_l[2:], label = r"$\eta$ = 0.01656")
    plt.plot(Epochs_s[3:], mse_test_s[3:], label = r"Decreasing $\eta$")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for SGD on Franke function with noise 0.5", size = 12)
    plt.legend()
    plt.show()

    print("Constant final MSE:")
    print(mse_test_h[-1])

    print("Changing final MSE:")
    print(mse_test_s[-1])




def momentum():
    constant_high = FrankeRegression(X_train, X_test, z_train, z_test)
    Epochs_h, mse_train_h, mse_test_h, r2_train_h, r2_test_h = constant_high.SGD(epochs = 1000, batch_size = 30, eta = 0.00481487*0.5, momentum = 0, lamb = 0.00127427, learning_schedule = "no", plot = "data")

    momentum_low = FrankeRegression(X_train, X_test, z_train, z_test)
    Epochs_l_m, mse_train_l, mse_test_l_m, r2_train_l, r2_test_l = momentum_low.SGD(epochs = 1000, batch_size = 30, eta = 0.00481487*0.5, momentum = 0.2, lamb = 0.00127427, learning_schedule = "no", plot = "data")

    momentum_lower = FrankeRegression(X_train, X_test, z_train, z_test)
    Epochs_low_m, mse_train_l, mse_test_low_m, r2_train_l, r2_test_l = momentum_lower.SGD(epochs = 1000, batch_size = 30, eta = 0.00481487*0.5, momentum = 0.5, lamb = 0.00127427, learning_schedule = "no", plot = "data")



    plt.plot(Epochs_h[4:], mse_test_h[4:], label = r"$\eta$ = 0.002407, without momentum")
    plt.plot(Epochs_l_m[4:], mse_test_l_m[4:], label = r"$\eta$ = 0.002407, momentum = 0.2")
    plt.plot(Epochs_low_m[4:], mse_test_low_m[4:], label = r"$\eta$ = 0.002407, momentum = 0.5")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for SGD on Franke function with noise 0.5", size = 12)
    plt.legend()
    plt.show()

    print("Constant final MSE:")
    print(mse_test_h[-1])

    print("Momentum 0.5 final MSE:")
    print(mse_test_l_m[-1])

    print("Momentum 0.8 final MSE:")
    print(mse_test_low_m[-1])




def gridsearch():
    #gridsearch for lambda and eta, here for SGD
    eta_min = np.log10(1e-5)   #log base 10
    eta_max = np.log10(0.025)    #upper limit
    eta_n = 20
    eta = np.logspace(eta_min, eta_max, eta_n)

    lamb_min = np.log10(1e-5)   #log base 10
    lamb_max = np.log10(1)   #upper limit
    lamb_n = 20


    lamb = np.logspace(lamb_min, lamb_max, lamb_n)

    mse_results = np.zeros((len(lamb), len(eta)))   #each row corresponds to one value of lambda, each column to a value of eta
    r2_results = np.zeros((len(lamb), len(eta)))


    for e in range(len(eta)):
        for l in range(len(lamb)):
            np.random.seed(123)
            SGD = FrankeRegression(X_train, X_test, z_train, z_test)
            z_model, z_predict = SGD.SGD(1000, eta = eta[e], batch_size = 30, momentum = 0, lamb = lamb[l])
            mse_train, mse_test, r2_train, r2_test = SGD.error(z_model, z_predict)
            mse_results[l, e] = mse_test  #row l, column e
            r2_results[l, e] = r2_test

            print(e, l)


    min = np.min(mse_results)
    index = np.where(mse_results == min)
    print("Poly 7")
    print("Min MSE: ", min)
    print("Min eta: ", eta[index[1]])
    print("Min lambda: ", lamb[index[0]])


    eta = np.round(np.log10(eta), 3)
    lamb = np.round(np.log10(lamb), 3)

    scale = 3.5


    plt.figure(figsize = (4*scale, 4*scale))
    ax_mse = sns.heatmap(mse_results, xticklabels = eta, yticklabels = lamb,  annot=True, cmap="YlGnBu", fmt='.3g')

    ax_mse.set_title("Test MSE for SGD on Franke function after 500 epochs", size = 20)
    ax_mse.set_xlabel(r"log$_{10}\eta$", size = 20)
    ax_mse.set_ylabel(r"log$_{10}\lambda$", size = 20)


    plt.show()


def batchloop():

    batches = np.linspace(1, 50, 50)

    MSE_test = []
    R2_test = []

    for b in batches:
        SGD = FrankeRegression(X_train, X_test, z_train, z_test)
        #z_model, z_predict = SGD.SGD(1000, eta = 0.00481487, batch_size = b, momentum = 0, lamb = 0.00127427)
        z_model, z_predict = SGD.SGD(epochs = 500, eta = 0.01656155, batch_size = b, momentum = 0, lamb = 0.0078476)
        mse_train, mse_test, r2_train, r2_test = SGD.error(z_model, z_predict)
        MSE_test.append(mse_test)
        R2_test.append(r2_test)
        print(b)


    min = np.min(MSE_test)
    index = np.where(MSE_test == min)
    print("Min MSE: ", min)
    print("Batch size: ", batches[index])

    plt.plot(batches, MSE_test)
    plt.xlabel("Batches", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE as a function of batch-size", size = 12)
    plt.legend()
    plt.show()



    plt.plot(batches, R2_test)
    plt.xlabel("Batches size", size = 12)
    plt.ylabel(r"Test R$^2$", size = 12)
    plt.title(r"Test R$^2$ score as a function of batch-size", size = 12)
    plt.legend()
    plt.show()



SGD_test()
#learning_rate()
#momentum()
#OLS()
#batchloop()
#gridsearch()
