import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

""" Grid search for optimal params """
learning_rates = np.logspace(-4, 1, 11)  # 11
lambdas = np.logspace(-5, 1, 20)         # 20

n_features = len(X_train[0])
y_train = y_train.reshape(-1, 1)#np.array([y_train]).reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
max_iterations = 10000

sklr_filename = f"part_e/sk_loglreg_{len(lambdas)}_{max_iterations}"
gd_filename = f"part_e/man_gd_{len(learning_rates)}_{len(lambdas)}_{max_iterations}"
sksgd_filename = f"part_e/sk_sgd_{len(learning_rates)}_{len(lambdas)}_{max_iterations}"
mansgd_filename = f"part_e/man_sgd_{len(learning_rates)}_{len(lambdas)}_{max_iterations}"


if train:

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
            print(f"Iteration: lambda {l+1}/{len(lambdas)} eta {e+1}/{len(learning_rates)}")
            beta = 0.01 * np.random.randn(n_features, 1)    # Initial random parameter inizialization

            """ Gradient Descent """
            for i in range(max_iterations):
                z = X_train_s.dot(beta)
                W = np.diag( (p(z)*(1-p(z))).reshape(-1) )

                grad_C = 1/len(X_train) * X_train_s.T @ (y_train - p(z)) + lambd * beta
                Hess_C = X_train_s.T @ W @ X_train_s

                #beta = beta + np.linalg.pinv(Hess_C) @ grad_C
                beta = beta + eta * grad_C      # Const learn_rate

            prob = p(X_test_s.dot(beta))

            pred = prob.round()

            score = np.sum(pred == y_test)/len(y_test)
            gd_score[e, l] = score

            """ Stochastic Gradient Descent """
            #sklearn

            SGDLogReg = SGDClassifier(loss = 'log', max_iter = max_iterations, alpha=lambd,\
                    learning_rate='constant', eta0 = eta, tol=1e-5)
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
                    grad = -(1/batch_size) * X_b.T @ (y_b - p(z)) + lambd * theta                    
                    if np.linalg.norm(grad) < tol:
                        break

                    theta = theta - eta*grad

            test_prob = p(X_test_s.dot(theta))

            man_pred = test_prob.round()
            man_sgd_score[e, l] = np.sum(man_pred==y_test)/len(y_test)

    np.savetxt(sklr_filename + ".txt", sk_logreg)
    np.savetxt(gd_filename + ".txt", gd_score)
    np.savetxt(sksgd_filename + ".txt", sk_sgd_score)
    np.savetxt(mansgd_filename + ".txt", man_sgd_score)

else:
    sk_logreg = np.loadtxt(sklr_filename + ".txt")
    gd_score = np.loadtxt(gd_filename + ".txt")
    sk_sgd_score = np.loadtxt(sksgd_filename + ".txt")
    man_sgd_score = np.loadtxt(mansgd_filename + ".txt")

""" Plot the results """
x = 3.5
xticks = [f"{i:.2e}" for i in lambdas]
yticks = [f"{i:.2e}" for i in learning_rates]
#print(xticks)
plt.figure(figsize = (4*x, 3*x))

sns.heatmap(gd_score, linewidth = 0.5, square = True, annot = True, cmap = 'YlGnBu', \
        cbar_kws={'shrink':0.6}, xticklabels=xticks, yticklabels=yticks)

plt.title(f"Accuracy for Gradient Descent {max_iterations} epochs")
plt.xlabel("$\lambda$", size = 20)
plt.ylabel("$\eta$", rotation = 0, size = 20)
plt.xticks(rotation = 45)
#plt.savefig(gd_filename + ".png")
plt.show()

plt.figure(figsize = (4*x, 3*x))

sns.heatmap(sk_sgd_score, linewidth = 0.5, square = True, annot = True, cmap = 'YlGnBu', \
        cbar_kws={'shrink':0.6}, xticklabels=xticks, yticklabels=yticks)

plt.title(f"Accuracy for Scikit-learns Stochastic Gradient Descent {max_iterations} epochs")
plt.xlabel("$\lambda$", size = 20)
plt.ylabel("$\eta$", rotation = 0, size = 20)
plt.xticks(rotation = 45)
#plt.savefig(sksgd_filename + ".png")
plt.show()

plt.figure(figsize = (4*x, 3*x))

sns.heatmap(man_sgd_score, linewidth = 0.5, square = True, annot = True, cmap = 'YlGnBu', \
        cbar_kws={'shrink':0.6}, xticklabels=xticks, yticklabels=yticks)

plt.title(f"Accuracy for Stochastic Gradient Descent {max_iterations} epochs")
plt.xlabel("$\lambda$", size = 20)
plt.ylabel("$\eta$", rotation = 0, size = 20)
plt.xticks(rotation = 45)
#plt.savefig(mansgd_filename + ".png")
plt.show()

fig, ax = plt.subplots()

plt.plot(lambdas, sk_logreg)

plt.title("Scikit-learns LogisticRegression\nAccuracy score as a function of $\lambda$")
plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
ax.set_xscale('log')
plt.ylim(0.94, 0.97)

#plt.savefig(sklr_filename + ".png")
plt.show()

""" Optimal params """
def get_opt_ind(matrix):
    ind = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    return ind

print(f"Epochs: {max_iterations}")

sk_sgd_ind = get_opt_ind(sk_sgd_score)
sksgd_e, sksgd_l = learning_rates[sk_sgd_ind[0]], lambdas[sk_sgd_ind[1]]
print(f"SGDClassifier: Highest score: {sk_sgd_score[sk_sgd_ind]} eta = {sksgd_e}, lamda = {sksgd_l}")

gd_ind = get_opt_ind(gd_score)
gd_e, gd_l = learning_rates[gd_ind[0]], lambdas[gd_ind[1]]
print(f"Gradient Descent: Highest score: {gd_score[gd_ind]} eta = {gd_e}, lamda = {gd_l}")

sgd_ind = get_opt_ind(man_sgd_score)
sgd_e, sgd_l = learning_rates[sgd_ind[0]], lambdas[sgd_ind[1]]
print(f"Stochastic Gradient Descent: Highest score: {man_sgd_score[sgd_ind]} eta = {sgd_e}, lamda = {sgd_l}")

sk_logreg_ind = np.argmax(sk_logreg)
print(f"LogisticRegression: Highest score: {np.max(sk_logreg)}, lambda = {lambdas[sk_logreg_ind]}")
