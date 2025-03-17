import numpy as np
import argparse
from dataset import load_data
from neural_network import NeuralNetwork

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Network")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use (default: mnist)")
    
    # Model Architecture
    parser.add_argument("--input_size", type=int, default=784, help="Input layer size (default: 784 for MNIST)")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes (default: [128, 64])")
    parser.add_argument("--output_size", type=int, default=10, help="Output layer size (default: 10 for MNIST)")
    
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    
    # Optimization
    parser.add_argument("--optimizer", type=str, choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer to use (default: adam)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum factor (default: 0.9, used for momentum & nesterov)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam (default: 0.9)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam (default: 0.999)")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Epsilon for Adam/Nadam (default: 1e-8)")
    
    # Activation & Weight Initialization
    parser.add_argument("--activation", type=str, choices=["sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function (default: ReLU)")
    parser.add_argument("--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method (default: random)")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log progress every N epochs (default: 10)")

    args = parser.parse_args()

    # Load dataset
    X_train, Y_train, X_test, Y_test = load_data(args.dataset)

    # Initialize model
    model = NeuralNetwork(
        input_size=args.input_size,
        hidden_layers=args.hidden_layers,
        output_size=args.output_size,
        activation=args.activation,
        weight_init=args.weight_init,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon
    )

    # Train model
    model.train(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size, log_interval=args.log_interval)

    # Evaluate model on test set
    Y_pred = model.predict(X_test)
    test_accuracy = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_test, axis=0))
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()