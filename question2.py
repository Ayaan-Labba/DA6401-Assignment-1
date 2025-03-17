import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=1, hidden_layers=[1], output_size=1, activation="ReLU", weight_init="random"):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.activation = activation
        self.weights, self.biases = self.init_weights(weight_init)

    def init_weights(self, weight_init):
        weights = []
        biases = []
        
        for i in range(len(self.layers) - 1):
            input_dim, output_dim = int(self.layers[i]), int(self.layers[i+1])
            
            if weight_init == "random":
                W = np.random.randn(output_dim, input_dim) * 0.01  # Small random values
            
            elif weight_init == "Xavier":
                W = np.random.randn(output_dim, input_dim) * np.sqrt(1 / input_dim)  # Xavier initialization
            
            else:
                raise ValueError("Invalid weight initialization method. Choose 'random' or 'Xavier'.")

            b = np.zeros((output_dim, 1))  # Initialize biases to zero

            weights.append(W)
            biases.append(b)
        
        return weights, biases

    def activation_function(self, x, derivative=False):
        if self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig) if derivative else sig
        elif self.activation == "tanh":
            tanh = np.tanh(x)
            return 1 - tanh ** 2 if derivative else tanh
        elif self.activation == "ReLU":
            return (x > 0) * 1 if derivative else np.maximum(0, x)
        else:
            return x  # Identity function

    def forward(self, X):
        self.h = [X]
        for i in range(len(self.weights)):
            a = np.dot(self.h[-1], self.weights[i].T) + self.biases[i].T
            self.h.append(self.activation_function(a))
        return self.h[-1]