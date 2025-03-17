import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=1, hidden_layers=[1], output_size=1, activation="ReLU", weight_init="random", 
                 optimizer="sgd", lr=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.activation = activation
        self.weights, self.biases = self.init_weights(weight_init)

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1  # Time step for Adam/Nadam

        # Optimizer state (for momentum-based optimizers)
        self.momentum_W = [np.zeros_like(w) for w in self.weights]
        self.momentum_B = [np.zeros_like(b) for b in self.biases]
        self.v_W = [np.zeros_like(w) for w in self.weights]  # Second moment for Adam/Nadam
        self.v_B = [np.zeros_like(b) for b in self.biases]

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
            return x if derivative else 1  # Identity function

    def forward(self, X):
        H, A = [X], []
        h = X

        for w, b in zip(self.weights, self.biases):
            a = np.dot(w, h) + b
            h = self.activation_function(a) if w is not self.weights[-1] else self.softmax(a)
            A.append(a)
            H.append(h)

        return A, H
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def loss(self, Y_pred, Y_true):
        m = Y_true.shape[1]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m
        return loss
    
    def backward(self, H, A, Y_true):
        m = Y_true.shape[1]
        da = -(Y_true - H[-1])  # Cross-entropy gradient w.r.t output layer
        dw, db = [], []

        for i in reversed(range(len(self.weights))):
            dw.insert(0, np.dot(da, H[i].T) / m)
            #db.insert(0, np.sum(da, axis=1, keepdims=True) / m)
            db.insert(0, np.da / m)
            if i > 0:
                dh = np.dot(self.weights[i].T, da)
                da = dh*self.activation_function(A[i-1], derivative=True)
                #da = np.dot(self.weights[i].T, da) * self.activation_function(A[i-1], derivative=True)

        return {"dw": dw, "db": db}
    
    def update_weights(self, grads):
        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.lr * grads["dw"][i]
                self.biases[i] -= self.lr * grads["db"][i]
            elif self.optimizer in ["momentum", "nesterov"]:
                self.momentum_W[i] = self.momentum * self.momentum_W[i] - self.lr * grads["dw"][i]
                self.momentum_B[i] = self.momentum * self.momentum_B[i] - self.lr * grads["db"][i]
                if self.optimizer == "nesterov":
                    self.weights[i] += self.momentum * self.momentum_W[i] - self.lr * grads["dw"][i]
                    self.biases[i] += self.momentum * self.momentum_B[i] - self.lr * grads["db"][i]
                else:
                    self.weights[i] += self.momentum_W[i]
                    self.biases[i] += self.momentum_B[i]
            elif self.optimizer in ["adam", "nadam"]:
                self.momentum_W[i] = self.beta1 * self.momentum_W[i] + (1 - self.beta1) * grads["dw"][i]
                self.momentum_B[i] = self.beta1 * self.momentum_B[i] + (1 - self.beta1) * grads["db"][i]
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (grads["dw"][i] ** 2)
                self.v_B[i] = self.beta2 * self.v_B[i] + (1 - self.beta2) * (grads["dB"][i] ** 2)

                m_W_corr = self.momentum_W[i] / (1 - self.beta1 ** self.t)
                m_B_corr = self.momentum_B[i] / (1 - self.beta1 ** self.t)
                v_W_corr = self.v_W[i] / (1 - self.beta2 ** self.t)
                v_B_corr = self.v_B[i] / (1 - self.beta2 ** self.t)

                if self.optimizer == "nadam":
                    m_W_corr = (self.beta1 * m_W_corr + (1 - self.beta1) * grads["dw"][i]) / (1 - self.beta1 ** self.t)
                    m_B_corr = (self.beta1 * m_B_corr + (1 - self.beta1) * grads["db"][i]) / (1 - self.beta1 ** self.t)

                self.weights[i] -= self.lr * m_W_corr / (np.sqrt(v_W_corr) + self.epsilon)
                self.biases[i] -= self.lr * m_B_corr / (np.sqrt(v_B_corr) + self.epsilon)

        self.t += 1  # Update time step for Adam/Nadam
