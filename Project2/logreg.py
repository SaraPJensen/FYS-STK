import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

np.random.seed(1337+420+69)

cancerdata = load_breast_cancer() #(569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = \
        train_test_split(data, target, test_size=0.2)

#print(X_train)
LogReg = LogisticRegression(max_iter = 10000)
LogReg.fit(X_train, y_train)

accuracy = LogReg.score(X_test, y_test)

print(f"Accuracy of sklearn LogisticRegresison: {accuracy:.5g}")


n_features = len(X_train[0])
beta = 0.0001 * np.random.randn(n_features, 1) 
#beta = LogReg.coef_.reshape(-1, 1)

def p(z):
    return 1/(1+np.exp(-z))
    #ee = np.exp(z)
    #return ee/(1+ee)

#print(z)
#print(p(z))
#print(p(z)*(1-p(z)))
#print(W)
y_train = np.array([y_train]).reshape(-1, 1)

eta = 0.0002
max_iterations = 100000
for i in range(max_iterations):
    #print("iteration: ", i)
    z = X_train.dot(beta)
    #print("z: ", np.shape(z))
    W = np.diag( (p(z)*(1-p(z))).reshape(-1) )
    #print("W: ", W.shape)

    #print("y train: ", y_train.shape)
    #print("p(z): ", p(z).shape)
    grad_C = 1/n_features * X_train.T @ (y_train - p(z))
    #print("Grad C: ", grad_C.shape)
    #Hess_C = - X_train.T @ W @ X_train
    #print("Hess C: ", Hess_C.shape)

    #beta = beta - np.linalg.pinv(Hess_C) @ grad_C
    beta = beta - eta * grad_C      # Const learn_rate
    #print("Beta: ", beta.shape)

print(LogReg.coef_)
print(beta)
