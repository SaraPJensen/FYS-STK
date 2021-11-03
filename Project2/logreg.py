import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

np.random.seed(1337+420+69)

cancerdata = load_breast_cancer() #(569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = \
        train_test_split(data, target, test_size=0.2)

#print(X_train)
LogReg = LogisticRegression(max_iter = 10000, penalty='none')
LogReg.fit(X_train, y_train)

accuracy = LogReg.score(X_test, y_test)

print(f"Accuracy of sklearn LogisticRegresison: {accuracy:.5g}")


n_features = len(X_train[0])
beta = 0.01 * np.random.randn(n_features, 1) 
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

""" Scaling our data """
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)
#y_train_s= scaler.fit_transform(y_train.reshape(-1, 1))
#y_test_s = scaler.fit_transform(y_test.reshape(-1, 1))

def out(arr):
    out = np.zeros(len(arr))
    for i, elem in enumerate(arr):
        if elem >= 0.5:
            out[i] = 1

    return out

eta = 0.0002
max_iterations = 10000
for i in range(max_iterations):
    #print("iteration: ", i)
    z = X_train_s.dot(beta)
    #print("z: ", np.shape(z))
    W = np.diag( (p(z)*(1-p(z))).reshape(-1) )
    #print("W: ", W.shape)

    #print("y train: ", y_train.shape)
    #print("p(z): ", p(z).shape)
    grad_C = 1/len(X_train) * X_train_s.T @ (y_train - p(z))
    #1/len(X_train) *
    #print("Grad C: ", grad_C.shape)
    Hess_C = X_train_s.T @ W @ X_train_s
    #print("Hess C: ", Hess_C.shape)

    #beta = beta + np.linalg.pinv(Hess_C) @ grad_C       # Bør det være + her? ref. Hastie
    beta = beta + eta * grad_C      # Const learn_rate, no overflow
    #print("Beta: ", beta.shape)

#print("Newton iteration")
#print(LogReg.coef_)
#print(beta.reshape(1, -1))
pred = out(p(X_test_s.dot(beta)))
print("Gradient Descent")
print(np.sum(pred == y_test),"/", len(y_test))

""" Stochastic Gradient Descent """
eta = 0.00025

#sklearn
SGDLogReg = SGDClassifier(loss = 'log', max_iter = 10000, alpha=0, learning_rate='constant', eta0 = eta)
SGDLogReg.fit(X_train, y_train)
SGDaccuracy = SGDLogReg.score(X_test, y_test)

print(f"Accuracy SGD Classifier: {SGDaccuracy}")




epochs = 10000
batch_size = 1
num_batches = int(n_features/batch_size) 
theta = np.random.randn(n_features, 1)

tol = 1e-6
for e in range(epochs):
    for b in range(num_batches):
        indeces = np.random.randint(0, high = n_features, size = batch_size)
        #print(indeces)

        X_b = X_train_s[indeces]
        #print(X_b.shape)
        y_b = y_train[indeces]

        #print(y_b)

        z = X_b.dot(theta)
        #grad = (1/batch_size) * X_b.T @ ((X_b @ theta) - y_b)
        grad = -(1/batch_size) * X_b.T @ (y_b - p(z))       # Funker med - (negativ)?
        #print(grad.shape)
        
        if np.linalg.norm(grad) < tol:
            break

        #print(theta.reshape(1, -1))
        theta = theta - eta*grad

#print("SGD")
#print(SGDLogReg.coef_)
#print(theta.reshape(1, -1))

test_prob = p(X_test_s.dot(theta))
#print(test_prob.reshape(1, -1))
#print(y_test)
#print(p(X_test_s.dot(SGDLogReg.coef_.reshape(-1, 1))).reshape(1, -1))
#test_prob = p(X_test_s.dot(LogReg.coef_.reshape(-1, 1)))
#print(test_prob) # Gir zeros
def out(arr):
    out = np.zeros(len(arr))
    for i, elem in enumerate(arr):
        if elem >= 0.5:
            out[i] = 1

    return out

man_ans = out(test_prob)
sk_ans = SGDLogReg.predict(X_test)#out(p(X_test_s.dot(SGDLogReg.coef_.reshape(-1, 1))))

#print(man_ans==sk_ans)
print("Sammenlikning, vår SGD vs SKLearn:")
print(f"{np.sum(man_ans==sk_ans)}/{len(man_ans)}")
#print("sk_ans:")
#print(sk_ans)
#print(sk_ans)
man_acc = np.sum(man_ans == y_test)
print("Johan II har da følgende treffsikkerhet på sin Logistiske Regresjonsanalyser:\n", man_acc, "/", len(y_test))
#print(man_ans)
#print(y_test)
#print(man_ans == y_test)
#print(sk_ans == y_test)

