from scalers import *
from functions import *
from sklearn.preprocessing import scale

MSE = mean_squared_error
R2 = r2_score

np.random.seed(2018)    # Same seed as functions.py

#Generate data

n = 20
x = np.sort((np.random.rand(n)))        # Equal to np.random.uniform(0, 1, n)
y = np.sort((np.random.rand(n)))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.1*np.random.randn(n, n)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)

# Manualy calculate the mse for a given polynomial and lambda by 
scaler = "standard"
poly = 10
k_fold = 5
reg_method = "Ridge"
n_lambda = 200

#manual_mse = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, n_lambda)

#print(np.shape(manual_mse))

def OLS_compare(x, y, z, scaler, lamb, poly):
    # Our function
    X = design_matrix(x, y, poly)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)
    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train,
                                                                z_test, scaler, lamb,
                                                                poly, plot="false")
    manual_mse = MSE(z_test_scaled, z_predict)
    
    # Scikit-learn method
    model = LinearRegression(fit_intercept = True)      # Fit intercept
    '''
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.fit_transform(X_test)
    z_train_sc = scaler.fit_transform(z_train.reshape(-1, 1))
    z_test_sc = scaler.fit_transform(z_test.reshape(-1, 1))
    X_train_sc = scale(X_train)
    X_test_sc = scale(X_test)
    z_train_sc = scale(z_train.reshape(-1, 1))
    z_test_sc = scale(z_test.reshape(-1, 1))
    '''
    #print(np.shape(X_train_sc))
    #print(np.shape(z_train_sc))
    model.fit(X_train, z_train)     # Fitter modellen uten skalering
    zpred = model.predict(X_test)

    sk_mse = MSE(z_test, zpred)     # Funker jævlig bra for z_test (uskalert), fordi modellen er fittet mot uskalert data. Funker SHIT når vi fitter mot skalert
    # Hvorfor funker det ikke :'(

    print("Manuell: ",manual_mse)
    print("sk: ", sk_mse)

OLS_compare(x_flat, y_flat, z_flat, "standard", 0, 8)

