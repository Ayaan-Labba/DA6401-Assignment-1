import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=1, hidden_layers=[1], output_size=1, activation="ReLU", weight_init="random", optimizer="sgd", 
                 lr=0.01, momentum=0.9, beta = 0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.activation = activation
        self.weights, self.biases = self.init_weights(weight_init)
        self.weight_decay = weight_decay

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1  # Time step for Adam/Nadam

        # Optimizer state (for momentum-based optimizers)
        self.u_w = [np.zeros_like(w) for w in self.weights]
        self.u_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]  # Second moment for Adam/Nadam
        self.v_b = [np.zeros_like(b) for b in self.biases]

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
    
    def gradients(self, H, A, Y_true):
        m = Y_true.shape[1]
        da = -(Y_true - H[-1])  # Cross-entropy gradient w.r.t output layer
        dw, db = [], []

        for i in reversed(range(len(self.weights))):
            dw.insert(0, np.dot(da, H[i].T) / m)  + self.weight_decay * self.weights[i]
            #db.insert(0, np.sum(da, axis=1, keepdims=True) / m)
            db.insert(0, np.da / m)
            if i > 0:
                dh = np.dot(self.weights[i].T, da)
                da = dh * self.activation_function(A[i-1], derivative=True)
                #da = np.dot(self.weights[i].T, da) * self.activation_function(A[i-1], derivative=True)

        return {"dw": dw, "db": db}
    
    def gradients_nag(self, H, A, Y_true, u_w):
        m = Y_true.shape[1]
        da = -(Y_true - H[-1])  # Cross-entropy gradient w.r.t output layer
        dw, db = [], []

        for i in reversed(range(len(self.weights))):
            dw.insert(0, np.dot(da, H[i].T) / m)  + self.weight_decay * self.weights[i]
            #db.insert(0, np.sum(da, axis=1, keepdims=True) / m)
            db.insert(0, np.da / m)
            if i > 0:
                dh = np.dot((self.weights[i] - u_w).T, da)
                da = dh * self.activation_function(A[i-1], derivative=True)
                #da = np.dot(self.weights[i].T, da) * self.activation_function(A[i-1], derivative=True)

        return {"dw": dw, "db": db}
    
    def gradient_descent(self, grads, grads_nag):
        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.lr * grads["dw"][i]
                self.biases[i] -= self.lr * grads["db"][i]
            
            elif self.optimizer in ["momentum", "nesterov"]:
                if self.optimizer == "nesterov":
                    self.u_w[i] += self.momentum * self.u_w[i] + grads_nag["dw"][i]
                    self.u_b[i] += self.momentum * self.u_b[i] + grads_nag["db"][i]
                else:
                    self.u_w[i] += self.momentum * self.u_w[i] + grads["dw"][i]
                    self.u_b[i] += self.momentum * self.u_b[i] + grads["db"][i]
                self.weights[i] -= self.lr * self.u_w[i]
                self.biases[i] -= self.lr * self.u_b[i]
            
            elif self.optimizer == "rmsprop":
                self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * grads["dw"][i] ** 2
                self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * grads["db"][i] ** 2
                self.weights[i] -= self.lr * grads["dw"][i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                self.biases[i] -= self.lr * grads["db"][i] / (np.sqrt(self.v_b[i]) + self.epsilon)
            
            elif self.optimizer in ["adam", "nadam"]:
                self.u_w[i] = self.beta1 * self.u_w[i] + (1 - self.beta1) * grads["dw"][i]
                self.u_b[i] = self.beta1 * self.u_b[i] + (1 - self.beta1) * grads["db"][i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads["dw"][i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads["dB"][i] ** 2)

                m_w_corr = self.u_w[i] / (1 - self.beta1 ** self.t)
                m_b_corr = self.u_b[i] / (1 - self.beta1 ** self.t)
                v_w_corr = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)

                if self.optimizer == "nadam":
                    m_w_corr = (self.beta1 * m_w_corr + (1 - self.beta1) * grads["dw"][i]) / (1 - self.beta1 ** self.t)
                    m_b_corr = (self.beta1 * m_b_corr + (1 - self.beta1) * grads["db"][i]) / (1 - self.beta1 ** self.t)

                self.weights[i] -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
                self.biases[i] -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

        self.t += 1  # Update time step for Adam/Nadam
    
    def train(self, X_train, Y_train, epochs, batch_size, log_interval=10):
        m = X_train.shape[1]
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_train, Y_train = X_train[:, indices], Y_train[:, indices]

            for i in range(0, m, batch_size):
                X_batch, Y_batch = X_train[:, i:i+batch_size], Y_train[:, i:i+batch_size]
                activations, pre_activations = self.forward(X_batch)
                grads = self.gradients(activations, pre_activations, Y_batch)
                self.gradient_descent(grads)

            if epoch % log_interval == 0:
                Y_pred = self.predict(X_train)
                loss = self.loss(Y_pred, Y_train)
                accuracy = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_train, axis=0))
                print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
