from neural_new import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore', category = ConvergenceWarning)

np.random.seed(1826)
'''
Sigmoid + Xavier:
    Epochs = 100

Relu & Leaky Relu + He:
    Epochs = 50
'''

sx_etas = np.logspace(-4, 1, 11)
rh_etas = np.logspace(-6, 0, 11)

lambdas = np.logspace(-5, 1, 20)

sx_eta = sx_etas[5]
sx_lamb = lambdas[0]

sk_sx_eta = sx_etas[2]
sk_sx_lamb = lambdas[0]

rh_eta = rh_etas[2]
rh_lamb = lambdas[0]

sk_rh_eta = rh_etas[6]
sk_rh_lamb = lambdas[8]

lrh_eta = rh_etas[2]
lrh_lamb = rh_etas[6]

""" Get cancer data """
cancer = load_breast_cancer()

data = cancer.data
target = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
y_test = y_test.reshape(-1, 1)

# Scale data
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

'''
# Scores
sx = []
rh = []

opt_sx = [(sx_eta, sx_lamb), (sk_sx_eta, sk_sx_lamb)]
opt_rh = [(rh_eta, rh_lamb), (lrh_eta, lrh_lamb), (sk_rh_eta, sk_rh_lamb)]

layers = [30, 30]

for eta, lambd in opt_sx:
    network = network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                                    layers, 100, 1, eta, lambd,\
                                    'sigmoid', cost_func='accuracy', dataset='classification',\
                                    weight_init_method=wi)
'''

def get_score_acc(af, wi, epochs, eta, lambd, layers = [30, 30]):
    network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
            layers, epochs, 1, eta, lambd, af, cost_func = 'accuracy',\
            dataset='classification', weight_init_method=wi)
    trash, score = network.model_training("SGD", plot="yes")

    return score

def get_sk_score_acc(af, epochs, eta, lambd, layers = (30, 30)):           # Weight initialization is not an option.
    network = MLPClassifier(hidden_layer_sizes = layers, activation=af, solver='sgd', alpha=lambd,\
            batch_size=1, learning_rate_init=eta, max_iter=epochs)

    score = []
    classes = np.unique(target)
    for ep in range(epochs):
        network.partial_fit(X_train_s, y_train, classes)
        
        score.append(network.score(X_test_s, y_test))

    return score


def score_epoch():
    #print("Starter sk-learn")
    sk_sx_score = get_sk_score_acc('logistic', 100, sx_eta, sx_lamb)
    #print("Starter custom")
    sx_score = get_score_acc('sigmoid', 'xavier', 100, sx_eta, sx_lamb)

    plt.plot(np.arange(1, 100+1), sx_score, label = "Sigmoid")
    plt.plot(np.arange(1, 100+1), sk_sx_score, label = "Sigmoid\nMLPDClassifier")
    plt.title("Accuracy as a function of epochs")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.savefig("part_d/sigmoid_compare.png")
    plt.show()

    rh_score = get_score_acc('relu', 'he', 50, rh_eta, rh_lamb)
    lrh_score = get_score_acc('leaky_relu', 'he', 50, lrh_eta, lrh_lamb)
    sk_rh_score = get_sk_score_acc('relu', 50, rh_eta, rh_lamb)

    plt.plot(np.arange(1, 50+1), rh_score, label = "ReLU")
    plt.plot(np.arange(1, 50+1), lrh_score, label = "Leaky ReLU")
    plt.plot(np.arange(1, 50+1), sk_rh_score, label = "ReLU\nMLPClassifier")
    plt.title("Accuracy as a function of epochs")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.savefig("part_d/relu_compare.png")
    plt.show()

#score_epoch()

def quick_score_acc(af, wi, epochs, eta, lambd, layers: list):
    if wi:
        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                layers, epochs, 1, eta, lambd, af, cost_func = 'accuracy',\
                dataset='classification', weight_init_method=wi)

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

    else:
        network = MLPClassifier(hidden_layer_sizes = tuple(layers), activation=af, solver='sgd', alpha=lambd,\
                batch_size=1, learning_rate_init=eta, max_iter=epochs)
        network.fit(X_train_s, y_train)
        acc = network.score(X_test_s, y_test)

    return acc


nodes = np.linspace(5, 50, 10)
nodes = nodes.astype(int)

sig1 = np.zeros((2, len(nodes)))
sig2 = np.zeros((2, len(nodes)))
sig3 = np.zeros((2, len(nodes)))

rel1 = np.zeros((2, len(nodes)))
rel2 = np.zeros((2, len(nodes)))
rel3 = np.zeros((2, len(nodes)))

lrel1 = np.zeros(len(nodes))
lrel2 = np.zeros(len(nodes))
lrel3 = np.zeros(len(nodes))

if sys.argv[1] == 'sigmoid' and sys.argv[2] == 'train':
    for i, num in enumerate(nodes):
        print(f"[{i+1}/{len(nodes)}]")
        #num = int(num)
        #ettlag, tolag, trelag = l1[i], l2[i], l3[i]
        '''
        sig1[0, i] = quick_score_acc('sigmod', 'xavier', 100, sx_eta, sx_lamb, [num])
        sig2[0, i] = quick_score_acc('sigmod', 'xavier', 100, sx_eta, sx_lamb, [num, num])
        sig3[0, i] = quick_score_acc('sigmod', 'xavier', 100, sx_eta, sx_lamb, [num, num, num])
        print("WTF")
        exit(1)
        '''
        #print("Jævla svinejævler")
        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num], 100, 1, sx_eta, sx_lamb, 'sigmoid', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='xavier')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        sig1[0, i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num], 100, 1, sx_eta, sx_lamb, 'sigmoid', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='xavier')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        sig2[0, i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num, num], 100, 1, sx_eta, sx_lamb, 'sigmoid', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='xavier')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        sig3[0, i] = acc

        sig1[1, i] = quick_score_acc('logistic', False, 100, sx_eta, sx_lamb, [num])
        sig2[1, i] = quick_score_acc('logistic', False, 100, sx_eta, sx_lamb, [num, num])
        sig3[1, i] = quick_score_acc('logistic', False, 100, sx_eta, sx_lamb, [num, num, num])

    np.savetxt("part_d/sig1.txt", sig1)
    np.savetxt("part_d/sig2.txt", sig2)
    np.savetxt("part_d/sig3.txt", sig3)

else:
    sig1 = np.loadtxt("part_d/sig1.txt")
    sig2 = np.loadtxt("part_d/sig2.txt")
    sig3 = np.loadtxt("part_d/sig3.txt")

if sys.argv[1] == 'relu' and sys.argv[2] == 'train':
    for i, num in enumerate(nodes):
        print(f"[{i+1}/{len(nodes)}]")

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num], 50, 1, rh_eta, rh_lamb, 'relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        rel1[0, i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num], 50, 1, rh_eta, rh_lamb, 'relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        rel2[0, i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num, num], 50, 1, rh_eta, rh_lamb, 'relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        rel3[0, i] = acc

        rel1[1, i] = quick_score_acc('logistic', False, 50, sx_eta, sx_lamb, [num])
        rel2[1, i] = quick_score_acc('logistic', False, 50, sx_eta, sx_lamb, [num, num])
        rel3[1, i] = quick_score_acc('logistic', False, 50, sx_eta, sx_lamb, [num, num, num])

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num], 50, 1, lrh_eta, lrh_lamb, 'leaky_relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        lrel1[i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num], 50, 1, lrh_eta, lrh_lamb, 'leaky_relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        lrel2[i] = acc

        network = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
                [num, num, num], 50, 1, lrh_eta,lrh_lamb, 'leaky_relu', cost_func = 'accuracy',\
                dataset='classification', weight_init_method='he')

        network.model_training("SGD", plot='no')

        prob = network.prediction(X_test_s)
        pred = prob.round()
        acc = np.sum(pred == y_test)/y_test.shape[0]

        lrel3[i] = acc

    np.savetxt("part_d/rel1.txt", rel1)
    np.savetxt("part_d/rel2.txt", rel2)
    np.savetxt("part_d/rel3.txt", rel3)

    np.savetxt("part_d/lrel1.txt", lrel1)
    np.savetxt("part_d/lrel2.txt", lrel2)
    np.savetxt("part_d/lrel3.txt", lrel3)

else:
    rel1 = np.loadtxt("part_d/rel1.txt")
    rel2 = np.loadtxt("part_d/rel2.txt")
    rel3 = np.loadtxt("part_d/rel3.txt")

    lrel1 = np.loadtxt("part_d/lrel1.txt")
    lrel2 = np.loadtxt("part_d/lrel2.txt")
    lrel3 = np.loadtxt("part_d/lrel3.txt")

if sys.argv[1] == 'sigmoid':
    plt.figure()
    plt.title("Accuracy as a function of nodes/layer\nOur FFNN; Sigmoid & Xavier")
    plt.plot(nodes, sig1[0], '.', label = "1 hidden layer", c='tab:blue')
    plt.plot(nodes, sig2[0], '.', label = "2 hidden layer", c='tab:orange')
    plt.plot(nodes, sig3[0], '.', label = "3 hidden layer", c='tab:green')
    
    plt.plot(nodes, sig1[0], alpha = 0.15)
    plt.plot(nodes, sig2[0], alpha = 0.15)
    plt.plot(nodes, sig3[0], alpha = 0.15)

    plt.xlabel("Nodes pr. layer")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("part_d/sig_epoch.png")

    plt.figure()
    plt.title("Accuracy as a function of nodes/layer\nScikit-learn MLPClassifier; Sigmoid")
    plt.plot(nodes, sig1[1], '.', label = "1 hidden layer", c='tab:blue')
    plt.plot(nodes, sig2[1], '.', label = "2 hidden layer", c='tab:orange')
    plt.plot(nodes, sig3[1], '.', label = "3 hidden layer", c='tab:green')
    
    plt.plot(nodes, sig1[1], alpha = 0.15)
    plt.plot(nodes, sig2[1], alpha = 0.15)
    plt.plot(nodes, sig3[1], alpha = 0.15)

    plt.xlabel("Nodes pr. layer")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("part_d/sk_sig_epoch.png")

if sys.argv[1] == 'relu':
    plt.figure()

    plt.title("Accuracy as a function of nodes/layer\nOur FFNN; ReLu & He")
    plt.plot(nodes, rel1[0], '.', label = "1 hidden layer", c='tab:blue')
    plt.plot(nodes, rel2[0], '.', label = "2 hidden layer", c='tab:orange')
    plt.plot(nodes, rel3[0], '.', label = "3 hidden layer", c='tab:green')

    plt.plot(nodes, rel1[0], alpha = 0.15)
    plt.plot(nodes, rel2[0], alpha = 0.15)
    plt.plot(nodes, rel3[0], alpha = 0.15)

    plt.xlabel("Nodes pr. layer")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("part_d/relu_epoch.png")

    plt.figure()

    plt.title("Accuracy as a function of nodes/layer\nScikit-learn MLPClassifier; ReLu")
    plt.plot(nodes, rel1[1], '.', label = "1 hidden layer", c='tab:blue')
    plt.plot(nodes, rel2[1], '.', label = "2 hidden layer", c='tab:orange')
    plt.plot(nodes, rel3[1], '.', label = "3 hidden layer", c='tab:green')

    plt.plot(nodes, rel1[1], alpha = 0.15)
    plt.plot(nodes, rel2[1], alpha = 0.15)
    plt.plot(nodes, rel3[1], alpha = 0.15)

    plt.xlabel("Nodes pr. layer")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("part_d/sk_relu_epoch.png")

    plt.figure()

    plt.title("Accuracy as a function of nodes/layer\nOur FFNN; Leaky ReLu & He")
    plt.plot(nodes, lrel1, '.', label = "1 hidden layer", c='tab:blue')
    plt.plot(nodes, lrel2, '.', label = "2 hidden layer", c='tab:orange')
    plt.plot(nodes, lrel3, '.', label = "3 hidden layer", c='tab:green')

    plt.plot(nodes, lrel1, alpha = 0.15)
    plt.plot(nodes, lrel2, alpha = 0.15)
    plt.plot(nodes, lrel3, alpha = 0.15)

    plt.xlabel("Nodes pr. layer")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("part_d/lrelu_epoch.png")
plt.show()

''' Test
np.random.seed(1826)
ass = NeuralNetwork(X_train_s, y_train, X_test_s, y_test,\
        [30], 100, 1, sx_eta, sx_lamb, 'sigmoid', cost_func = 'accuracy',\
        dataset='classification', weight_init_method='xavier')

ass.model_training("SGD", plot='no')

prob = ass.prediction(X_test_s)
pred = prob.round()
acc = np.sum( pred == y_test )/y_test.shape[0]
'''
