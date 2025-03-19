import wandb
import train
import sys

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization method (more efficient than grid or random)
    'metric': {
        'name': 'val_accuracy',  # Optimize for validation accuracy
        'goal': 'maximize'       # We want to maximize accuracy
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'num_layers': {
            'values': [3, 4, 5]
        },
        'hidden_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0.0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [0.001, 0.0001]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_init': {
            'values': ['random', 'Xavier']
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'ReLU']
        }
    }
}

def sweep_train():
    # Initialize a new wandb run
    run = wandb.init()
    
    # Access all hyperparameters through wandb.config
    config = wandb.config
    
    # Generate a descriptive name for this run
    run_name = f"hl_{config.num_layers}_sz_{config.hidden_size}_bs_{config.batch_size}_ac_{config.activation}_opt_{config.optimizer}_w_{config.weight_init}_lr_{config.learning_rate}_wd_{config.weight_decay}"
    wandb.run.name = run_name
    wandb.run.save()
    
    # Construct command to run train.py with appropriate parameters
    sys.argv = [
        "train.py",
        "--wandb_entity", wandb.run.entity,
        "--wandb_project", wandb.run.project,
        "--epochs", str(config.epochs),
        "--num_layers", str(config.num_layers),
        "--hidden_size", str(config.hidden_size),
        "--weight_decay", str(config.weight_decay),
        "--learning_rate", str(config.learning_rate),
        "--optimizer", config.optimizer,
        "--batch_size", str(config.batch_size),
        "--weight_init", config.weight_init,
        "--activation", config.activation
    ]
    
    # Add specific optimizer parameters based on which optimizer is selected
    if config.optimizer in ['momentum', 'nag']:
        sys.argv.extend(["--momentum", "0.9"])
    elif config.optimizer == 'rmsprop':
        sys.argv.extend(["--beta", "0.9"])
    elif config.optimizer in ['adam', 'nadam']:
        sys.argv.extend(["--beta1", "0.9", "--beta2", "0.999"])
    
    # Run the training script as a subprocess with the current hyperparameters
    train.main()
    
    # Finish the wandb run
    wandb.finish()

# Initialize the sweep
def main():
    # Set up wandb project and entity
    wandb_entity = "ch21b021-indian-institute-of-technology-madras"  # Replace with your wandb entity name
    wandb_project = "DA6401-Assignment-1"  # Replace with your project name
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)

    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_train, count=30)  # Run 30 experiments

if __name__ == "__main__":
    main()