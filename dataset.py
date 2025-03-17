from keras.datasets import fashion_mnist
from keras.datasets import mnist

# Load dataset
def load_data(dataset="fashion_mnist"):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
        return X_train, y_train, X_test, y_test
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
    return X_train, y_train, X_test, y_test

