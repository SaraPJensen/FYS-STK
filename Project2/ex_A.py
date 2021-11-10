from franke_regclass import *



X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 30, noise = 0.05, poly=8)



GD = FrankeRegression(X_train, X_test, z_train, z_test)
z_model, z_predict = GD.GD(1000, eta = 0, momentum = 0, lamb = 0)
print("Gradient descent error:")
GD.print_error(z_model, z_predict)

print()

SGD = FrankeRegression(X_train, X_test, z_train, z_test)
z_model, z_predict = SGD.SGD(1000, eta = 0.0025, batch_size = 30, momentum = 0, lamb = 0)
print("Stochastic gradient descent error:")
SGD.print_error(z_model, z_predict)

print()

OLS = FrankeRegression(X_train, X_test, z_train, z_test)
z_model, z_predict = OLS.OLS_Ridge(lamb = 0)
print("OLS error:")
OLS.print_error(z_model, z_predict)

print()

Ridge = FrankeRegression(X_train, X_test, z_train, z_test)
z_model, z_predict = Ridge.OLS_Ridge(lamb = 0.05)
print("Ridge error:")
Ridge.print_error(z_model, z_predict)






#gridsearch for lambda and eta, here for SGD
eta_min = -7   #log base 10
eta_max = -2
eta_n = 2
eta = np.logspace(eta_min, eta_max, eta_n)

lamb_min = -7   #log base 10
lamb_max = -2
lamb_n = 2


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
print("Min MSE: ", min)
print("Min eta: ", eta[index[1]])
print("Min lambda: ", lamb[index[0]])


eta = np.round(np.log10(eta), 3)
lamb = np.round(np.log10(lamb), 3)

ax_mse = sns.heatmap(mse_results, xticklabels = eta, yticklabels = lamb,  annot=True, cmap="YlGnBu", fmt='.4g')

ax_mse.set_title("MSE")
ax_mse.set_xlabel(r"log10$\eta$")
ax_mse.set_ylabel(r"log10$\lambda$")

plt.show()
