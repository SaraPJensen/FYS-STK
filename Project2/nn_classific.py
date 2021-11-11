from neural_new import *
import sys
import seaborn as sns

train = False       # Joao magic
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

wi = 'xavier'           # weight_initialization, same variance for each layer
af = 'sigmoid'          # activation_function
epochs = 100
batch_size = 1          # Same as sk-learn

learning_rates = np.logspace(-4, 1, 11)
lambdas = np.logspace(-5, 1, 20)

accs = np.zeros((len(learning_rates), len(lambdas)))  # Accuracies
target = y_test.reshape(-1, 1)
#print(target.shape)
filename = f'cv_nn_class_{str(wi[0])}_{str(af[0])}_{epochs}.txt'

if train:
    for i, eta in enumerate(learning_rates):
        for j, lambd in enumerate(lambdas):
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
    
    np.savetxt(filename, accs)
else:
    accs = np.loadtxt(filename)

""" Plot the accuracies """
sns.heatmap(accs, linewidth = 0.5, square = True, cmap = 'YlGnBu')
#x, y = np.meshgrid(learning_rates, lambdas)
#plt.pcolormesh(x, y, accs)
plt.xlabel("$\lambda$")
plt.ylabel("$\eta$")

#print(accs)
plt.show()
