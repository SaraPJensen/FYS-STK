#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from functions import *

np.random.seed(2018)

#Generate the data
n = 20
noise = 0.5
x = np.arange(0,1,1/n)
y = np.arange(0,1,1/n)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + noise*np.random.randn(n, n)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)


poly = 5
scaler = "none"
lamb = 0

#Make design matrix and split the data
X = design_matrix(x_flat, y_flat, poly)
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

#Plot the graph
ThreeD_plot(x, y, z, "Franke function with low noise")

#Plot prediction and calculate errors
z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

mse_train = mean_squared_error(z_train_scaled, z_model)
print(f"MSE, train: {mse_train:.5}")
print('')
mse_test = mean_squared_error(z_test_scaled, z_predict)
print(f"MSE, test: {mse_test:.5}")
print('')

r2_train = r2_score(z_train_scaled, z_model)
print(f"R2, train: {r2_train:.5}")
print('')
r2_test = r2_score(z_test_scaled, z_predict)
print(f"R2, test: {r2_test:.5}")
print('')


beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
k = len(beta)       # number of parameters

ste_beta = np.zeros(k)
width = np.zeros(k) # Width of the confidence interval
a = 0.10            # 100(1-a)% CI
n = len(X_train)    # Number of samples

for i in range(k):
    ste_beta[i] = noise * np.sqrt((np.linalg.pinv(X_train.T @ X_train))[i, i])
    width[i] = ste_beta[i] * 1.6499     # t-value for a = 0.1, df = 304 # n - (k+1)

plt.scatter(np.arange(len(beta)), beta)
plt.errorbar(np.arange(len(beta)), beta, xerr = 0, yerr = width, linestyle='')

plt.title("90% Confidence Intervals for $\\beta_i$\nwith noise = " + str(noise) + "$\\mathcal{N}$(0, 1)")
plt.xlabel("$i$")
plt.ylabel("$\hat{\\beta}_i \pm t_{\\alpha/2, n-(k+1)} \cdot s_{\hat{\\beta}_i}$")
plt.savefig(f"90CI_poly5_noise{noise}.png")

plt.show()