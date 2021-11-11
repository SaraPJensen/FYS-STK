from neural_new import *

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
epochs = 5000
batch_size = 1          # Same as sk-learn

learning_rates = np.logspace(-4, 1, 11)
lambdas = np.logspace(-5, 1, 20)

accs = np.zeros(len(learning_rates), len(lambdas))  # Accuracies

for eta in learning_rates:
    for lambd in lambdas:
        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                                layers, epochs, batch_size, eta, lambd,\
                                af, cost_func='accuracy', dataset='classification',\
                                weight_init_method=wi)
        network.model_training("SGD", plot='no')
