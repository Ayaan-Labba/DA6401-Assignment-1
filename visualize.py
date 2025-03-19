import matplotlib.pyplot as plt
import numpy as np
from dataset import load_data
import wandb

# Display one sample from each class
def visualize_samples():
    # Initialize wandb
    wandb.init(project="DA6401-Assignment-1", entity="ch21b021-indian-institute-of-technology-madras", name="Data Visualization")

    X_train, y_train, _, _ = load_data(dataset="fashion_mnist")
    class_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Fashion-MNIST Class Samples", fontsize=18, fontweight='bold')
    indices = [np.where(y_train == i)[0][0] for i in range(10)]
    for ax, idx, label in zip(axes.flat, indices, class_labels):
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    # Save the figure locally
    plt.savefig('fashion_mnist_samples.png')
    plt.close()
    
    # Log the image to wandb
    wandb.log({"fashion_mnist_samples": wandb.Image('fashion_mnist_samples.png')})
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    visualize_samples()