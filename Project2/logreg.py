import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

np.random.seed(1337+420+69)
#warnings.filterwarnings("ignore") # Disable warning outprint, use with caution
warnings.filterwarnings("ignore", category = ConvergenceWarning)

train = False
if len(sys.argv) - 1:
    train = sys.argv[1]

cancerdata = load_breast_cancer() # shape: (569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = \
        train_test_split(data, target, test_size=0.2)

""" Scaling our data """
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

def p(z):       # Sigmoid / Logistic function
    return 1/(1+np.exp(-z))

#if train:
""" Grid search for optimal params """
learning_rates = np.logspace(-4, 1, 2)  # 11
lambdas = np.logspace(-5, 1, 2)         # 20

n_features = len(X_train[0])
y_train = y_train.reshape(-1, 1)#np.array([y_train]).reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
max_iterations = 10000

sk_logreg = np.zeros(len(lambdas))
gd_score = np.zeros((len(learning_rates), len(lambdas)))
sk_sgd_score = np.zeros((len(learning_rates), len(lambdas)))
man_sgd_score = np.zeros((len(learning_rates), len(lambdas)))

for l, lambd in enumerate(lambdas):
    """ Sk LogReg """ # only depends on lambda
    ''' By default the regression is used with the lBFGS
        (limited-memory Broyden-Fletcher-Goldfarb-Shanno) method,
        a quasi-Newton method. Newton-Cotes however is an option.
    '''
    LogReg = LogisticRegression(max_iter = max_iterations, penalty='l2', C = 1/lambd,\
            multi_class = 'multinomial')
    LogReg.fit(X_train, y_train.ravel())

    sk_logreg[l] = LogReg.score(X_test, y_test)
    
    for e, eta in enumerate(learning_rates):

        beta = 0.01 * np.random.randn(n_features, 1)    # Initial random parameter inizialization
#beta = LogReg.coef_.reshape(-1, 1)

#ee = np.exp(z)
#return ee/(1+ee)

#print(z)
#print(p(z))
#print(p(z)*(1-p(z)))
#print(W)

#y_train_s= scaler.fit_transform(y_train.reshape(-1, 1))
#y_test_s = scaler.fit_transform(y_test.reshape(-1, 1))


        """ Gradient Descent """
        for i in range(max_iterations):
            #print("iteration: ", i)
            z = X_train_s.dot(beta)
            #print("z: ", np.shape(z))
            W = np.diag( (p(z)*(1-p(z))).reshape(-1) )
            #print("W: ", W.shape)

            #print("y train: ", y_train.shape)
            #print("p(z): ", p(z).shape)
            grad_C = 1/len(X_train) * X_train_s.T @ (y_train - p(z)) + lambd * beta
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
        #pred = out(p(X_test_s.dot(beta)))
        prob = p(X_test_s.dot(beta))
        #print(prob)
        pred = prob.round()

        score = np.sum(pred == y_test)/len(y_test)
        gd_score[e, l] = score

        """ Stochastic Gradient Descent """
        #sklearn

        SGDLogReg = SGDClassifier(loss = 'log', max_iter = max_iterations, alpha=lambd,\
                learning_rate='constant', eta0 = eta)
        SGDLogReg.fit(X_train, y_train.ravel())
        SGDaccuracy = SGDLogReg.score(X_test, y_test)

        sk_sgd_score[e, l] = SGDaccuracy


        epochs = 10000
        batch_size = 1
        num_batches = int(n_features/batch_size) 
        theta = np.random.randn(n_features, 1)

        tol = 1e-6
        for ep in range(epochs):
            for b in range(num_batches):
                indeces = np.random.randint(0, high = n_features, size = batch_size)

                X_b = X_train_s[indeces]
                y_b = y_train[indeces]


                z = X_b.dot(theta)
                #grad = (1/batch_size) * X_b.T @ ((X_b @ theta) - y_b)
                grad = -(1/batch_size) * X_b.T @ (y_b - p(z)) + lambd * theta# Funker med - (negativ)?
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
        man_pred = test_prob.round()
        man_sgd_score[e, l] = np.sum(man_pred==y_test)/len(y_test)

        #man_sgd_score = test_prob.round()

print(f"Sk-learn Logistic Regression:\n{sk_logreg}")
print(f"Gradient descent:\n{gd_score}")
print(f"Sk-learn SGD:\n{sk_sgd_score}")
print(f"Manual SGD:\n{man_sgd_score}")

'''
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
'''
