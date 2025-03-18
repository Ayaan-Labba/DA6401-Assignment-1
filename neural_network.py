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
                w = np.random.rand(output_dim, input_dim) # Small random values
            
            elif weight_init == "Xavier":
                w = np.random.rand(output_dim, input_dim) * np.sqrt(1 / input_dim)  # Xavier initialization

            b = np.zeros((output_dim, 1))  # Initialize biases to zero

            weights.append(w)
            biases.append(b)
        
        return weights, biases

    def activation_function(self, x, derivative=False):
        #x = np.clip(x, -500, 500)
        if self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig) if derivative else sig
        elif self.activation == "tanh":
            tanh = np.tanh(x)
            return 1 - tanh ** 2 if derivative else tanh
        elif self.activation == "ReLU":
            return (x > 0) * 1 if derivative else np.maximum(0, x)
        else:
            return 1 if derivative else x  # Identity function

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        H, A = [X], []
        h = X

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = np.dot(h, w.T) + b.T
            #a = np.clip(a, -1e10, 1e10)
            A.append(a)

            if i < len(self.weights) - 1:
                h = self.activation_function(a)
            else:
                # Apply softmax for the output layer
                h = self.softmax(a)
            
            H.append(h)
            
        return H, A
    
    def softmax(self, x):
        #x = np.clip(x, -500, 500)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def loss(self, Y_pred, Y_true):
        n = Y_true.shape[0]
        #Y_pred = np.clip(Y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(Y_true * np.log(Y_pred + self.epsilon)) / n
        return loss
    
    def gradients(self, H, A, Y_true):
        n = Y_true.shape[0]
        da = -(Y_true - H[-1])  # Cross-entropy gradient w.r.t output layer
        dw, db = [], []

        for i in reversed(range(len(self.weights))):
            dw.insert(0, np.dot(H[i].T, da) / n) #+ self.weight_decay * self.weights[i]
            db.insert(0, np.sum(da, axis=0, keepdims=True) / n)
            if i > 0:
                dh = np.dot(da, self.weights[i])
                #dh = np.clip(dh, -1e10, 1e10)
                da = dh * self.activation_function(A[i-1], derivative=True)
                #print(np.max(A[i-1]))
                #da = np.clip(da, -1e10, 1e10)

        return {"dw": dw, "db": db}

    def gradients_nag(self, H, A, Y_true):
        n = Y_true.shape[0]
        da = -(Y_true - H[-1])  # Cross-entropy gradient w.r.t output layer
        dw, db = [], []

        for i in reversed(range(len(self.weights))):
            dw.insert(0, np.dot(H[i].T, da) / n) #+ self.weight_decay * self.weights[i]
            db.insert(0, np.sum(da, axis=0, keepdims=True) / n)
            if i > 0:
                dh = np.dot(da, self.weights[i] - self.u_w[i])
                #dh = np.clip(dh, -1e10, 1e10)
                da = dh * self.activation_function(A[i-1], derivative=True)
                #print(np.max(A[i-1]))
                #da = np.clip(da, -1e10, 1e10)

        return {"dw": dw, "db": db}
    
    def gradient_descent(self, grads, grads_nag=None):
        for i in range(len(self.weights)):
            if self.optimizer == "sgd":
                self.weights[i] -= self.lr * grads["dw"][i].T
                self.biases[i] -= self.lr * grads["db"][i].T
            
            elif self.optimizer in ["momentum", "nesterov"]:
                # Use appropriate gradients based on optimizer
                if self.optimizer == "nesterov" and grads_nag is not None:
                    grad_w = grads_nag["dw"][i].T
                    grad_b = grads_nag["db"][i].T
                else:
                    grad_w = grads["dw"][i].T
                    grad_b = grads["db"][i].T
                
                # Clip gradients to prevent explosion
                grad_w = np.clip(grad_w, -5.0, 5.0)
                grad_b = np.clip(grad_b, -5.0, 5.0)
                
                # Update velocities with decay
                self.u_w[i] = self.momentum * self.u_w[i] + self.lr * grad_w
                self.u_b[i] = self.momentum * self.u_b[i] + self.lr * grad_b
                
                # Clip velocities to prevent explosion
                self.u_w[i] = np.clip(self.u_w[i], -25.0, 25.0)
                self.u_b[i] = np.clip(self.u_b[i], -25.0, 25.0)
                
                self.weights[i] -= self.u_w[i]
                self.biases[i] -= self.u_b[i]
            
            elif self.optimizer == "rmsprop":
                self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (grads["dw"][i] ** 2).T
                self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (grads["db"][i] ** 2).T
                self.weights[i] -= self.lr * grads["dw"][i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                self.biases[i] -= self.lr * grads["db"][i] / (np.sqrt(self.v_b[i]) + self.epsilon)
            
            elif self.optimizer in ["adam", "nadam"]:
                self.u_w[i] = self.beta1 * self.u_w[i] + (1 - self.beta1) * grads["dw"][i].T
                self.u_b[i] = self.beta1 * self.u_b[i] + (1 - self.beta1) * grads["db"][i].T
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads["dw"][i] ** 2).T
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads["db"][i] ** 2).T

                m_w_corr = self.u_w[i] / (1 - self.beta1 ** self.t)
                m_b_corr = self.u_b[i] / (1 - self.beta1 ** self.t)
                v_w_corr = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)

                if self.optimizer == "nadam":
                    m_w_corr = (self.beta1 * m_w_corr + (1 - self.beta1) * grads["dw"][i]).T / (1 - self.beta1 ** self.t)
                    m_b_corr = (self.beta1 * m_b_corr + (1 - self.beta1) * grads["db"][i]).T / (1 - self.beta1 ** self.t)

                self.weights[i] -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
                self.biases[i] -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

        self.t += 1  # Update time step for Adam/Nadam
    
    def train(self, X_train, Y_train, epochs, batch_size, log_interval=1):
        n = X_train.shape[0]
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                X_batch, Y_batch = X_train[i:i+batch_size], Y_train[i:i+batch_size]
                H, A = self.forward(X_batch)
                if self.optimizer == "nesterov":
                    grads_nag = self.gradients_nag(H, A, Y_batch)
                grads = self.gradients(H, A, Y_batch)
                self.gradient_descent(grads)

            if epoch % log_interval == 0:
                Y_pred = self.predict(X_train)
                loss = self.loss(Y_pred, Y_train)
                #accuracy = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_train, axis=0))
                print(f"Epoch {epoch}: Loss={loss:.4f}")

    def predict(self, X):
        """
        Predicts the class probabilities for input X.
        """
        activations, _ = self.forward(X)
        return activations[-1]
