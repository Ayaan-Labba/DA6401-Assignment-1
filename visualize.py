import matplotlib.pyplot as plt
import numpy as np
from dataset import load_data

# Display one sample from each class
def visualize_samples():
    (X_train, y_train), _ = load_data(dataset="fashion_mnist")
    class_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Fashion-MNIST Class Samples", fontsize=18, fontweight='bold')
    indices = [np.where(y_train == i)[0][0] for i in range(10)]
    for ax, idx, label in zip(axes.flat, indices, class_labels):
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.show()