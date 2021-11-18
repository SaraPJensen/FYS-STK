from neural_new import *
import sys
import warnings

from sklearn.neural_network import MLPClassifier        # For comparison
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category = ConvergenceWarning)

train = False
if len(sys.argv) - 1:
    train = sys.argv[1]

# 'Optimal parameters', at least sufficiantly good
""" Get cancer data """
cancer = load_breast_cancer()

data = cancer.data
target = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

# Scale data
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

""" Initialize network with 'optimal' parameters """

layers = [30, 30]       # Two layers, 30 nodes each

wi = 'he'           # weight_initialization, same variance for each layer
af = 'relu'          # activation_function
epochs = 50
batch_size = 1          # Same as sk-learn

# Use this with sigmoid + xavier
#learning_rates = np.logspace(-4, 1, 11) #Various learning rates and lambda values
# Use this with relu/leaky relu + He
learning_rates = np.logspace(-6, 0, 11) #Various learning rates and lambda values
lambdas = np.logspace(-5, 1, 20)

accs = np.zeros((len(learning_rates), len(lambdas)))      # Accuracies
sk_accs = np.zeros((len(learning_rates), len(lambdas)))   # Accuracies 

target = y_test.reshape(-1, 1)      # Same shape as the prediction

filename = f'nn_class_{str(wi[0])}_{str(af[0])}_{epochs}'    # Used for saving
skfilename = 'sk_'+filename
filename = 'part_d/' + filename
skfilename = 'part_d/' + skfilename

if train:
    for i, eta in enumerate(learning_rates):
        for j, lambd in enumerate(lambdas):
            """ Custom method """
            network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                                    layers, epochs, batch_size, eta, lambd,\
                                    af, cost_func='accuracy', dataset='classification',\
                                    weight_init_method=wi)
            print(f"Initialized model with\neta [{i+1}/{len(learning_rates)}] = {eta}\nlambda [{j+1}/{len(lambdas)}] = {lambd}")
            network.model_training("SGD", plot='no')

            prob = network.prediction(X_test_s)
            pred = prob.round()
            acc = np.sum(pred == target)/target.shape[0]

            accs[i, j] = acc

            '''
            """ sk-learn """ # No leaky_relu
            sk_network = MLPClassifier(hidden_layer_sizes = tuple(layers), activation='logistic',\
                    solver='sgd', alpha=lambd, batch_size=batch_size, learning_rate_init=eta,\
                    max_iter=epochs)

            sk_network.fit(X_train_s, y_train)

            sk_acc = sk_network.score(X_test_s, target)
            sk_accs[i, j] = sk_acc
            '''
    
    np.savetxt(filename + '.txt', accs)
    np.savetxt(skfilename + '.txt', sk_accs)

else:
    accs = np.loadtxt(filename + '.txt')
    sk_accs = np.loadtxt(skfilename + '.txt')

""" Plot the accuracies """
''' # Or not
x = 3.5
xlabels = [f"{i:.2g}" for i in lambdas]
ylabels = [f"{i:.2g}" for i in learning_rates]

plt.figure(figsize=(4*x, 3*x))
hm = sns.heatmap(accs, linewidth = 0.5, square = True, annot = True, cmap = 'YlGnBu',\
        cbar_kws={'shrink':0.6}, xticklabels=xlabels, yticklabels=ylabels)
plt.xlabel("$\lambda$")
plt.ylabel("$\eta$", rotation = 0)
plt.xticks(rotation = 45)
plt.title(f"Accuracy for custom network, {epochs} epochs")
plt.savefig(filename+".png")
plt.show()

plt.figure(figsize=(4*x, 3*x))
sns.heatmap(sk_accs, linewidth = 0.5, square = True, annot = True, cmap = 'YlGnBu',\
        cbar_kws={'shrink':0.6}, xticklabels=xlabels, yticklabels=ylabels)
plt.xlabel("$\lambda$")
plt.ylabel("$\eta$", rotation = 0)
plt.xticks(rotation = 45)
plt.title(f"Accuracy for sk-learn network, {epochs} epochs")
plt.savefig(skfilename+".png")
plt.show()
#'''

print(f"Epochs: {epochs}")
print("Custom:")
ind = np.unravel_index(np.argmax(accs, axis=None), accs.shape)
eind = ind[0]
lind = ind[1]
print(f"best accuracy: {np.max(accs)}\nparams: eta [{eind}/{len(learning_rates)}] = {learning_rates[eind]}, lambda [{lind}/{len(lambdas)}] = {lambdas[lind]}")

print("\nSk-learn:")
skind = np.unravel_index(np.argmax(sk_accs, axis=None), sk_accs.shape)
skeind = skind[0]
sklind = skind[1]
print(f"best accuracy: {np.max(sk_accs)}\nparams: eta [{skeind}/{len(learning_rates)}] = {learning_rates[skeind]}, lambda [{sklind}/{len(lambdas)}] = {lambdas[sklind]}")

"""
#Prediction Accuracy for optimal parameters and custom epochs, only for testing
eps = 50
# Note that the penalization is now set to 0
network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test, layers, eps, batch_size, learning_rates[eind], lambdas[lind],\
                    af, cost_func='accuracy', dataset='classification', weight_init_method=wi)
network.model_training("SGD", plot='no')

prob = network.prediction(X_test_s)
pred = prob.round()
acc = np.sum(pred == target)/target.shape[0]

print(f"Accuracy for optimal parameters and {eps} epochs")
print(f"{np.sum(pred==target)}/{target.shape[0]} = {acc}")
"""
