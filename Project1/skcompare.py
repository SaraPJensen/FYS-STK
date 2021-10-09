from scalers import *
from functions import *
from sklearn.preprocessing import scale

MSE = mean_squared_error
R2 = r2_score

np.random.seed(2018)    # Same seed as functions.py

#Generate data

n = 20
x = np.random.rand(n)        # Equal to np.random.uniform(0, 1, n)
y = np.random.rand(n)
x, y = np.meshgrid(x, y)
noise = 0.15
z = FrankeFunction(x, y) + noise * np.random.randn(n, n)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)

# Manualy calculate the mse for a given polynomial and lambda by 
scaler = "none"
poly = 10
k_fold = 5
reg_method = "Ridge"
n_lambda = 200

X = design_matrix(x_flat, y_flat, poly)
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size = 0.2)

#manual_mse = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, n_lambda)

#print(np.shape(manual_mse))

def OLS_compare(X_train, X_test, z_train, z_test, scaler, poly):
    # Our function
    n = int((p+1) * (p+2)/2)
    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train[:, :n], X_test[:, :n], z_train,
                                                                z_test, scaler, 0,
                                                                poly, plot="false")
    man_mse = MSE(z_test_scaled, z_predict)
    man_r2 = R2(z_test_scaled, z_predict)
    
    # Scikit-learn method
    model = LinearRegression(fit_intercept = False)      # Fit intercept
    '''
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.fit_transform(X_test)
    z_train_sc = scaler.fit_transform(z_train.reshape(-1, 1))
    z_test_sc = scaler.fit_transform(z_test.reshape(-1, 1))
    '''
    '''
    X_train_sc = scale(X_train)
    X_test_sc = scale(X_test)
    z_train_sc = scale(z_train.reshape(-1, 1))
    z_test_sc = scale(z_test.reshape(-1, 1))
    '''
    model.fit(X_train[:, :n], z_train)     # Fitter modellen uten skalering
    zpred = model.predict(X_test[:, :n])

    #model.fit(X_train_sc, z_train_sc)     # Fitter modellen med skalering
    #zpred_sc = model.predict(X_test_sc)

    sk_mse = MSE(z_test, zpred)     # Funker jævlig bra for z_test (uskalert), fordi modellen er fittet mot uskalert data. Funker SHIT når vi fitter mot skalert
    sk_r2 = R2(z_test, zpred)
    #print(f"\nPoly: {poly}")
    #print("Manuell: ",man_mse)
    #print("sk: ", sk_mse)
    return man_mse, sk_mse, man_r2, sk_r2

def ridge_compare(x, y, z, scaler, lamb, poly):
    
    X = design_matrix(x, y, poly)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train,
                                                                z_test, scaler, lamb,
                                                                poly, plot="false")
    man_mse = MSE(z_test_scaled, z_predict)

    model = linear_model.Ridge(alpha = lamb, fit_intercept = False)
    model.fit(X_train, z_train)
    zpred = model.predict(X_test)

    sk_mse = MSE(z_test, zpred)

    return man_mse, sk_mse

def lasso_compare(x, y, z, scaler, lamb, poly):

    X = design_matrix(x, y, poly)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    z_train_scaled, z_test_scaled, z_predict, z_model = Lasso(X_train, X_test, z_train,
                                                                z_test, scaler, lamb,
                                                                poly, plot="false")
    man_mse = MSE(z_test_scaled, z_predict)

    model = linear_model.Lasso(alpha = lamb, fit_intercept = False)
    model.fit(X_train, z_train)
    zpred = model.predict(X_test)

    sk_mse = MSE(z_test, zpred)

    return man_mse, sk_mse


### MAIN ###

## OLS comparison
'''
man_ols_mse = np.zeros(poly)
sk_ols_mse = np.zeros(poly)
man_ols_r2 = np.zeros(poly)
sk_ols_r2 = np.zeros(poly)

for p in range(1, poly + 1):
    man_mse, sk_mse, man_r2, sk_r2 = OLS_compare(X_train, X_test, z_train, z_test, scaler, p)

    man_ols_mse[p-1] = man_mse
    sk_ols_mse[p-1] = sk_mse
    man_ols_r2[p-1] = man_r2
    sk_ols_r2[p-1] = sk_r2

print("OLS:")
print("Polygrad, Man mse, sk mse, man r2, sk r2")
for i in range(poly):
    print(f"{i+1} {man_ols_mse[i]:5.5g} {sk_ols_mse[i]:5.5g} {man_ols_r2[i]:5.5g} {sk_ols_r2[i]:5.5g}")
#OLS_compare(x_flat, y_flat, z_flat, "standard", 8)
'''

## Ridge comparison

ridge_man = np.zeros((5, 5))
ridge_sk = np.zeros((5, 5))
#lasso_man = np.zeros((5, 5))
#lasso_sk = np.zeros((5, 5))

ls = np.logspace(-4, 1, 5)
for p in range(5):              # Husk å bruke en fornuftig scaler
    for l in range(len(ls)):
        man_mse, sk_mse = ridge_compare(x_flat, y_flat, z_flat, scaler, ls[l], p)
        ridge_man[p, l] = man_mse
        ridge_sk[p, l] = sk_mse
        '''
        man_mse, sk_mse = lasso_compare(x_flat, y_flat, z_flat, scaler, ls[l], p)
        lasso_man[p, l] = man_mse
        lasso_sk[p, l] = sk_mse
        '''

print(ridge_man)
print(ridge_sk)          # Denne sammenlikningen gir lignende resultater men sklearn er en størrelsesorden bedre. DETTE KOMMER AV SKALERING!! MÅ ALDRI SKALERE!! IKKE MED STANDARD I ALLE FALL
#print(lasso_man)
#print(lasso_sk)
