import numpy as np
import argparse
from dataset import load_data
from neural_network import NeuralNetwork
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def preprocess_data(X_train, Y_train, X_test, Y_test):
    X_train_reshape = np.array(X_train).reshape(X_train.shape[0], -1).astype(float)
    X_test_reshape = np.array(X_test).reshape(X_test.shape[0], -1).astype(float)

    Y_train_reshape = np.zeros((Y_train.shape[0], 10)).astype(float)
    Y_test_reshape = np.zeros((Y_test.shape[0], 10)).astype(float)

    for i in range(Y_train.shape[0]):
        y = Y_train[i]
        Y_train_reshape[i, y] = 1
    
    for i in range(Y_test.shape[0]):
        y = Y_test[i]
        Y_test_reshape[i, y] = 1
    
    return X_train_reshape, Y_train_reshape, X_test_reshape, Y_test_reshape

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
    parser = argparse.ArgumentParser(description="Train a Neural Network")
    
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use for training")
    
    # Wandb pr
    parser.add_argument('-wp', '--wandb_project', default="DA6401-Assignment-1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', default="ch21b021-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")

    # Model Architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help="Number of hidden neurons in a feedforward layer")
    
    # Training Parameters
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function for training")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help="Weight decay used by optimizers")
    
    # Optimization
    parser.add_argument('-o', '--optimizer', type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer to use for updating weights")
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help="Epsilon used by optimizers")
    
    # Activation & Weight Initialization
    parser.add_argument('-a', '--activation', type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function for each layer")
    parser.add_argument('-w_i', '--weight_init', type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method ('random' preffered for 'ReLU' activation, 'Xavier' preferred for 'sigmoid' and 'tanh' activations)")
    
    # Logging
    parser.add_argument('-l_i', '--log_interval', type=int, default=1, help="Log progress every N epochs")

    args = parser.parse_args()

    run_name = f"hl_{args.num_layers}_bs_{args.batch_size}_ac_{args.activation}_opt_{args.optimizer}"

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name)

    # Load dataset
    X_train, Y_train, X_test, Y_test = load_data(args.dataset)

    X_train_reshape, Y_train_reshape, X_test_reshape, Y_test_reshape = preprocess_data(X_train, Y_train, X_test, Y_test)

    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(X_train_reshape, Y_train_reshape, test_size=0.1, random_state=42)
    
    # Create model with hidden layers of the same size
    hidden_layers = [args.hidden_size] * args.num_layers

    # Initialize model
    model = NeuralNetwork(
        input_size=x_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay
    )

    # Train the model
    history = model.train(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        log_interval=args.log_interval,
        wandb_log=True
    )

    # Log metrics to wandb
    wandb.log(history)

    # # Evaluate on test set
    # Y_pred = model.predict(X_test_reshape)
    # test_loss = model.loss(Y_pred, Y_test_reshape)
    # test_accuracy = model.accuracy(Y_pred, Y_test_reshape)

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # # Log metrics to wandb
    # wandb.log({
    #     "test_loss": test_loss,
    #     "test_accuracy": test_accuracy
    # })
    
    # # Create and log confusion matrix
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] if args.dataset == 'fashion_mnist' else None
    
    # cm_image = plot_confusion_matrix(Y_test_reshape, Y_pred, class_names)
    # wandb.log({"confusion_matrix": wandb.Image(cm_image)})
    
    # Finish wandb run
    wandb.finish()

    return model

if __name__ == "__main__":
    main()