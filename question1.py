from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Reshape and normalize data
X_train = X_train.reshape(-1, 28, 28) / 255.0

# Class names for Fashion-MNIST in the given order
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot an example image for each class
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Fashion-MNIST Class Samples", fontsize=18, fontweight='bold')

for i in range(10):
    idx = np.where(y_train == i)[0][0]  # Get the first index of each class
    axs[i // 5, i % 5].imshow(X_train[idx], cmap='gray')
    axs[i // 5, i % 5].set_title(class_names[i], pad=4, fontsize=12, fontweight='light')
    axs[i // 5, i % 5].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
