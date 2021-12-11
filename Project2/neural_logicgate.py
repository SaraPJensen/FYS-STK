from neural_new import *


x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
X_train = np.column_stack((x1, x2))
z_train = [0, 0, 0, 1]

z_train = np.array(z_train)


X_test = X_train
z_test = z_train

hidden_nodes = [2]   #This is a list of the number of nodes in each hidden layer
eta = 0.0001    #0.1 or 0.01 works well for sigmoid, same for relu and leaky_relu
batch_size = 4
epochs = 10000
lamb = 0

activation_func = "sigmoid"
cost_func = "accuracy"     #relu and leaky_relu only works with mse
dataset = "classification"
weight_init_method = "none"


Neural = NeuralNetwork(X_train, z_train, X_test, z_test, hidden_nodes, epochs, batch_size, eta, lamb, activation_func, cost_func, dataset, weight_init_method)
Neural.model_training("GD")

z_model = Neural.prediction(X_train)

print(z_model)


z_classified = classify(z_model)
results = np.column_stack((z_train, z_classified))
accuracy = np.abs(z_train.ravel() - z_classified.ravel())
total_wrong = sum(accuracy)
percentage = (len(accuracy) - total_wrong)/len(accuracy)

#print("Training results")
print(results)
print('')

'''
z_predict = Neural.prediction(X_test)
z_predict_class = classify(z_predict)
results_test = np.column_stack((z_test, z_predict_class))
accuracy_test = np.abs(z_test.ravel() - z_predict_class.ravel())
total_wrong_test = sum(accuracy_test)
percentage_test = (len(accuracy_test) - total_wrong_test)/len(accuracy_test)

'''
#print("Test results")
#print(results_test)

print("Train accuracy: ", percentage)

#print("Test accuracy: ", percentage_test)
