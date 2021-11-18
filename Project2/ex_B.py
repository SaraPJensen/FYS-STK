from neural_new import *
from sklearn.neural_network import MLPRegressor

#X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0.05, design = "stack")

X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0.05, design = "stack")


def benchmark_split(noise):
    X_train_noise, X_test_noise, z_train_noise, z_test_noise = Franke_data(n_dpoints = 20, noise = noise, design = "stack")
    X_train, X_test, z_train, z_test = Franke_data(n_dpoints = 20, noise = 0, design = "stack")


    print()
    print("Noise: ", noise)
    print("True MSE: ", mean_squared_error(z_test_noise, z_test))
    print("True R2: ", r2_score(z_test_noise, z_test))
    print()

#benchmark_split(0.05)



def test_Sigmoid():
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    eta = 0.01591789    #0.05 and 0.01 works well for leaky relu and relu, 0.03 works for sigmoid
    batch_size = 30
    epochs = 1000
    lamb = 0.00027826

    eta = 1
    lamb = 0


    activation_func = "sigmoid"
    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "xavier"

    np.random.seed(123)
    Neural = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    mse_train, mse_test, r2_train, r2_test = Neural.model_training(training_method, plot = "no")
    print(mse_test)
    print(r2_test)

    z_model = Neural.prediction(X_train)
    z_predict = Neural.prediction(X_test)

    results = np.column_stack((z_train, z_model))

    print()
    print("Train MSE: ", mean_squared_error(z_train, z_model))
    print("Test MSE: ", mean_squared_error(z_test, z_predict))
    print('')
    print("Train R2 score: ", r2_score(z_train, z_model))
    print("Test R2 score: ", r2_score(z_test, z_predict))
    print()


#test_Sigmoid()



def gridsearch():
    #Necessary parameters
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    batch_size = 30
    epochs = 1000

    activation_func = "sigmoid"
    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "nielsen"


    #gridsearch for lambda and eta
    eta_min = np.log10(0.0001)   #log base 10
    eta_max = np.log10(1)    #upper limit
    eta_n = 10
    eta = np.logspace(eta_min, eta_max, eta_n)


    lamb_min = np.log10(0.00001)   #log base 10
    lamb_max = np.log10(0.01)   #upper limit
    lamb_n = 10
    lamb = np.logspace(lamb_min, lamb_max, lamb_n)



    mse_results = np.zeros((len(lamb), len(eta)))   #each row corresponds to one value of lambda, each column to a value of eta
    r2_results = np.zeros((len(lamb), len(eta)))

    print()
    print()
    print()




    for e in range(len(eta)):
        for l in range(len(lamb)):
            np.random.seed(123)
            NN = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta[e], lamb[l], activation_func, cost_func, dataset, weight_init_method)
            mse_train, mse_test, r2_train, r2_test = NN.model_training(method = training_method, plot = "no")

            mse_results[l, e] = mse_test  #row l, column e
            r2_results[l, e] = r2_test

            print(e, l)
            print()


    print(weight_init_method)
    print(activation_func)
    print()
    min_mse = np.min(mse_results)
    index_mse = np.where(mse_results == min_mse)
    print("Min MSE (mse): ", min_mse)
    print("R2 (mse): ", r2_results[index_mse])
    print("Min eta (mse): ", eta[index_mse[1]])
    print("Min lambda (mse): ", lamb[index_mse[0]])
    print()

    max_r2 = np.max(r2_results)
    index_r2 = np.where(r2_results == max_r2)
    print("MSE (r2): ", mse_results[index_r2])
    print("Max R2 (r2): ", max_r2)
    print("Eta (r2): ", eta[index_r2[1]])
    print("Lambda (r2): ", lamb[index_r2[0]])




    eta = np.round(np.log10(eta), 3)
    lamb = np.round(np.log10(lamb), 3)

    scale = 3.5
    plt.figure(figsize = (4*scale, 4*scale))

    ax_mse = sns.heatmap(mse_results, xticklabels = eta, yticklabels = lamb,  annot=True, cmap="YlGnBu")

    ax_mse.set_title("MSE from FFNN with Sigmoid, noise = 0.05", size = 20)
    ax_mse.set_xlabel(r"log$_{10}(\eta)$", size = 20)
    ax_mse.set_ylabel(r"log$_{10}(\lambda)$", size = 20)

    plt.show()


gridsearch()



def nodes():
    l10 = [10, 10]
    l50 = [50, 50]
    l100 = [100, 100]

    batch_size = 30
    epochs = 1000
    eta = 1
    lamb = 0

    activation_func = "sigmoid"
    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "nielsen"

    np.random.seed(123)
    L10 = NeuralNetwork(X_train, z_train, X_test, z_test, l10, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, l10_mse_train, l10_mse_test, l10_r2_train, l10_r2_test = L10.model_training(training_method, "data")

    np.random.seed(123)
    L50 = NeuralNetwork(X_train, z_train, X_test, z_test, l50, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, l50_mse_train, l50_mse_test, l50_r2_train, l50_r2_test = L50.model_training(training_method, "data")

    np.random.seed(123)
    L100 = NeuralNetwork(X_train, z_train, X_test, z_test, l100, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, l100_mse_train, l100_mse_test, l100_r2_train, l100_r2_test = L100.model_training(training_method, "data")


    plt.plot(Epochs[3:], l10_mse_test[3:], label = "10 x 10 nodes")
    plt.plot(Epochs[3:], l50_mse_test[3:], label = "50 x 50 nodes")
    plt.plot(Epochs[3:], l100_mse_test[3:], label = "100 x 100 nodes")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for FFNN using Sigmoid with noise 0.5", size = 12)
    plt.legend()
    plt.show()

#nodes()



def activation():
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    eta = 0.5    #0.05 and 0.01 works well for leaky relu and relu, 0.03 works for sigmoid
    batch_size = 30
    epochs = 500
    lamb = 0

    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "nielsen"


    np.random.seed(123)
    activation_func = "sigmoid"
    eta = 0.027825594
    lamb = 0
    eta = 1
    sig = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, sig_mse_train, sig_mse_test, sig_r2_train, sig_r2_test = sig.model_training(training_method, "data")
    print("MSE sigmoid: ", sig_mse_test[-1])

    np.random.seed(123)
    activation_func = "relu"
    eta = 0.02154435
    lamb = 0
    eta = 1
    lamb = 0.00021544
    relu = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, relu_mse_test, _r2_train, relu_r2_test = relu.model_training(training_method, "data")
    print("MSE relu: ", relu_mse_test[-1])


    np.random.seed(123)
    activation_func = "leaky_relu"
    eta = 0.01591789
    lamb = 0.00027826
    eta = 1
    lamb = 0.00021544
    l_relu = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, l_relu_mse_test, _r2_train, l_relu_r2_test = l_relu.model_training(training_method, "data")
    print("MSE leaky relu: ", l_relu_mse_test[-1])


    plt.plot(Epochs[:], sig_mse_test[:], label = "Sigmoid")
    plt.plot(Epochs[:], relu_mse_test[:], label = "ReLU")
    plt.plot(Epochs[:], l_relu_mse_test[:], label = "Leaky ReLU")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for FFNN for different activation functions", size = 12)
    plt.legend()
    plt.show()



#activation()



def weights():
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    #0.05 and 0.01 works well for leaky relu and relu, 0.03 works for sigmoid
    batch_size = 30
    epochs = 500
    lamb = 0

    activation_func = "leaky_relu"
    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"


    np.random.seed(123)
    weight_init_method = "none"
    #eta = 0.27825594   #sigmoid
    #lamb = 0   #sigmoid
    #eta = 0.02154435   #relu
    #lamb = 0.0000359381   #relu
    eta = 0.01591789    #leaky relu
    lamb = 0.000027826   #leaky relu
    none = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, none_mse_test, _r2_train, none_r2_test = none.model_training(training_method, "data")
    print("Final none: ", none_mse_test[-1])


    np.random.seed(123)
    weight_init_method = "nielsen"
    eta = 1   #all
    lamb = 0.00021544   #relu and leaky relu
    #lamb = 0     #sigmoid
    nielsen = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, nielsen_mse_test, _r2_train, nielsen_r2_test = nielsen.model_training(training_method, "data")
    print("Final Nielsen: ", nielsen_mse_test[-1])

    '''
    np.random.seed(123)
    weight_init_method = "xavier"
    eta = 1
    eta= 0.05
    xavier = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, xavier_mse_test, _r2_train, xavier_r2_test = xavier.model_training(training_method, "data")
    print("Final xavier: ", xavier_mse_test[-1])
    '''

    eta = 1   #both
    #lamb = 0.00046416    #relu
    lamb = 0.00021544   #leaky relu
    np.random.seed(123)
    weight_init_method = "he"
    he = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, he_mse_test, _r2_train, he_r2_test = he.model_training(training_method, "data")
    print("Final he: ", he_mse_test[-1])



    plt.plot(Epochs[2:], none_mse_test[2:], label = "No scaling")
    plt.plot(Epochs[2:], nielsen_mse_test[2:], label = "Nielsen")
    #plt.plot(Epochs[:], xavier_mse_test[:], label = "Xavier")
    plt.plot(Epochs[2:], he_mse_test[2:], label = "He")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for FFNN for Leaky ReLU using different weight initialisations", size = 12)
    plt.legend()
    plt.show()

#weights()



def learning():
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    batch_size = 30
    epochs = 1000
    lamb = 0

    cost_func = "mse"
    activation_func = "sigmoid"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "nielsen"


    np.random.seed(123)
    eta = 0.5
    highest = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, highest_mse_test, _r2_train, highest_r2_test = highest.model_training(training_method, "data")
    print("MSE highest: ", highest_mse_test[-1])

    np.random.seed(123)
    eta = 0.1
    high = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, high_mse_test, _r2_train, high_r2_test = high.model_training(training_method, "data")
    print("MSE high: ", high_mse_test[-1])


    np.random.seed(123)
    eta = 0.05
    low_relu = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, low_mse_test, _r2_train, low_r2_test = low_relu.model_training(training_method, "data")
    print("MSE low: ", low_mse_test[-1])

    np.random.seed(123)
    eta = 0.01
    lowest = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
    Epochs, _mse_train, lowest_mse_test, _r2_train, lowest_r2_test = lowest.model_training(training_method, "data")
    print("MSE lowest: ", lowest_mse_test[-1])


    plt.plot(Epochs, highest_mse_test, label = r"$\eta$ = 0.5")
    plt.plot(Epochs, high_mse_test, label = r"$\eta$ = 0.1")
    plt.plot(Epochs, low_mse_test, label = r"$\eta$ = 0.05")
    plt.plot(Epochs, lowest_mse_test, label = r"$\eta$ = 0.01")
    plt.xlabel("Epochs", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for FFNN for different learning rates", size = 12)
    plt.legend()
    plt.show()



#learning()





def batchsize():
    batches = np.linspace(1, 100, 50)

    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    eta = 1    #0.05 and 0.01 works well for leaky relu and relu, 0.03 works for sigmoid
    batch_size = 30
    epochs = 1000
    lamb = 0

    activation_func = "sigmoid"
    cost_func = "mse"
    dataset = "function"
    training_method = "SGD"
    weight_init_method = "nielsen"

    MSE = []
    R2 = []


    for b in batches:
        np.random.seed(123)
        NN = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, b, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
        mse_train, mse_test, r2_train, r2_test = NN.model_training(method = training_method, plot = "no")

        MSE.append(mse_test)
        R2.append(r2_test)
        print(b)


    plt.plot(batches, MSE)
    plt.xlabel("Batchsize", size = 12)
    plt.ylabel("Test MSE", size = 12)
    plt.title(f"Test MSE for FFNN using Sigmoid with noise 0.05", size = 12)
    plt.legend()
    plt.show()

    plt.plot(batches, R2)
    plt.xlabel("Batchsize", size = 12)
    plt.ylabel("Test R2 score", size = 12)
    plt.title(f"Test R2 score for FFNN using Sigmoid with noise 0.05", size = 12)
    plt.legend()
    plt.show()


#batchsize()



def SKlearn():
    hidden_nodes = [50, 50]   #This is a list of the number of nodes in each hidden layer
    eta = 1
    batch_size = 30
    epochs = 1000
    lamb = 0.00027826

    eta = 1
    lamb = 0

    activation_func = "logistic"

    np.random.seed(123)
    sk_network = MLPRegressor(hidden_layer_sizes = tuple(hidden_nodes), activation = activation_func,\
            solver = "sgd", alpha = lamb, batch_size = batch_size, learning_rate = "constant", learning_rate_init=eta, \
            max_iter = epochs, momentum = 0)

    sk_network.fit(X_train, z_train)

    z_predict = sk_network.predict(X_test)

    print()
    print("Test MSE: ", mean_squared_error(z_test, z_predict))
    print('')
    print("Test R2 score: ", r2_score(z_test, z_predict))
    print()


#SKlearn()
