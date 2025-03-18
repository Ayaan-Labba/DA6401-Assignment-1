import wandb
import subprocess

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
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
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

# Initialize the sweep
def main():
    # Set up wandb project and entity
    wandb_entity = "ch21b021-indian-institute-of-technology-madras"  # Replace with your wandb entity name
    wandb_project = "DA6401-Assignment-1"  # Replace with your project name
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=wandb_project, entity=wandb_entity)
    
    # Define the sweep agent function that will run the training with different configs
    def sweep_agent():
        # Start a new wandb run
        run = wandb.init()
        
        # Get hyperparameters from wandb
        config = wandb.config
        
        # Construct command to run train.py with appropriate parameters
        cmd = [
            "python", "train.py",
            "--wandb_entity", wandb_entity,
            "--wandb_project", wandb_project,
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
            cmd.extend(["--momentum", "0.9"])
        elif config.optimizer == 'rmsprop':
            cmd.extend(["--beta", "0.9"])
        elif config.optimizer in ['adam', 'nadam']:
            cmd.extend(["--beta1", "0.9", "--beta2", "0.999"])
        
        # Run the training script as a subprocess with the current hyperparameters
        subprocess.run(cmd)
        
        # Finish the wandb run
        wandb.finish()
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_agent, count=30)  # Run 30 experiments

if __name__ == "__main__":
    main()