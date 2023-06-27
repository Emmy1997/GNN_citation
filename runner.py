import wandb
from train import train_and_evaluate


def sweep_entry():
    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "lr": {"values": [0.02]},
            "num_heads": {"values": [3]},
            "num_epochs": {"value": 400},
            "num_layers": {"values": [1]},
            "p": {"values": [0.4]}
        }
    }

    # Initialize Wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="GAT_citation")

    # Run the sweep
    wandb.agent(sweep_id, function=train_and_evaluate)

# Run the sweep entry function
sweep_entry()
