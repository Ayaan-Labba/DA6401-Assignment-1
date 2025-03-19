import numpy as np
import argparse
from dataset import load_data
from neural_network import NeuralNetwork
import wandb
from sklearn.model_selection import train_test_split

def preprocess_data(X, Y):
    X_reshape = np.array(X).reshape(X.shape[0], -1).astype(float)
    Y_reshape = np.zeros((Y.shape[0], 10)).astype(float)

    for i in range(Y.shape[0]):
        y = Y[i]
        Y_reshape[i, y] = 1
    
    return X_reshape, Y_reshape

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Network")
    
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use for training")
    
    # Wandb pr
    parser.add_argument('-wp', '--wandb_project', default="DA6401-Assignment-1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', default="ch21b021-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")

    # Model Architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help="Number of hidden neurons in a feedforward layer")
    
    # Training Parameters
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function for training")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate used to optimize model parameters")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.005, help="Weight decay used by optimizers")
    
    # Optimization
    parser.add_argument('-o', '--optimizer', type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="rmsprop", help="Optimizer to use for updating weights")
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help="Momentum used by momentum and nag optimizers")
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help="Beta used by rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help="Epsilon used by optimizers")
    
    # Activation & Weight Initialization
    parser.add_argument('-a', '--activation', type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU", help="Activation function for each layer")
    parser.add_argument('-w_i', '--weight_init', type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method ('random' preffered for 'ReLU' activation, 'Xavier' preferred for 'sigmoid' and 'tanh' activations)")
    
    # Logging
    parser.add_argument('-l_i', '--log_interval', type=int, default=1, help="Log progress every N epochs")

    args = parser.parse_args()

    run_name = f"hl_{args.num_layers}_sz_{args.hidden_size}_bs_{args.batch_size}_ac_{args.activation}_opt_{args.optimizer}_w_{args.weight_init}_lr_{args.learning_rate}_wd_{args.weight_decay}"

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name)

    # Load dataset
    X_train, Y_train, _, _ = load_data(args.dataset)

    X_train_reshape, Y_train_reshape = preprocess_data(X_train, Y_train)

    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(X_train_reshape, Y_train_reshape, test_size=0.1, random_state=42)
    
    # Create model with hidden layers of the same size
    hidden_layers = [args.hidden_size] * args.num_layers

    # Initialize model
    model = NeuralNetwork(
        input_size=x_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=y_train.shape[1],
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
    
    # Finish wandb run
    wandb.finish()

    return model

if __name__ == "__main__":
    main()