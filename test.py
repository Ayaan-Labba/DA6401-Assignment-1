from neural_network import NeuralNetwork
import numpy as np
from dataset import load_data

X_train, Y_train, X_test, Y_test = load_data()
X_train = np.array(X_train).reshape(X_train.shape[0], -1)
Y_train = np.array(Y_train).reshape(-1, 1)
model = NeuralNetwork(input_size=X_train.shape[1], output_size=10, hidden_layers=[32, 32], weight_init="Xavier", activation="sigmoid")
model.train(X_train, Y_train, epochs=5, batch_size=100)
