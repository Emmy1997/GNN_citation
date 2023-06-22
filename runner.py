import json
import wandb
from train import train_and_evaluate, load_wandb

def sweep_entry():
    # Define sweep configuration
    sweep_config = {
        "method": "random",
        "metric": {"name": "val\loss", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 0.0001, "max": 0.1},
            "hidden_channels": {"values": [16, 32, 64, 128]},
            "num_epochs": {"value": 350},
            "num_layers": {"values": [3,4,5,6]},
            "activation": {"values": ["relu", "tanh"]}
        }
    }

    # Initialize Wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="GNN_citation")

    def runner():
        # Define configuration variables
        config_defaults = {
            "lr": 0.01,
            "hidden_channels": 64,
            "num_epochs": 10,
            "num_layers": 3,
            "activation": "relu"
        }

        # Initialize Wandb run
        with wandb.init(config=config_defaults):
            # Get the current configuration
            config = wandb.config

            # Load dataset and other configurations
            load_wandb(config)

            # Perform training and evaluation
            train_and_evaluate(config)

    # Run the sweep
    wandb.agent(sweep_id, function=runner)

# Run the sweep entry function
sweep_entry()
