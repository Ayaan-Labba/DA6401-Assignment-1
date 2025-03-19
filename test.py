import numpy as np
from dataset import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import train
import wandb

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix."""
    # Convert from one-hot to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure for wandb
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return 'confusion_matrix.png'

def main():
    _, _, X_test, Y_test = load_data(dataset="fashion_mnist")
    x_test, y_test = train.preprocess_data(X_test, Y_test)
    
    # Evaluate on test set
    model = train.main()

    Y_pred = model.predict(x_test)
    test_loss = model.loss(Y_pred, y_test)
    test_accuracy = model.accuracy(Y_pred, y_test)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Log metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    run_name = "Test data confusion matrix"

    wandb.init(project="DA6401-Assignment-1", entity="ch21b021-indian-institute-of-technology-madras", name=run_name)
    
    # Create and log confusion matrix
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    cm_image = plot_confusion_matrix(y_test, Y_pred, class_names)
    wandb.log({"confusion_matrix": wandb.Image(cm_image)})

    # Finish wandb run
    wandb.finish()

    return Y_pred

if __name__ == "__main__":
    main()
