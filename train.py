# General imports
import json

# Data science imports
import plotly

# Import Weights & Biases for Experiment Tracking
import wandb

# Graph imports
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


import networkx as nx
from tqdm.auto import trange
from visualize import GraphVisualization

# internal files
from models import GCN
from dataset import HW3Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_wandb = True #@param {type:"boolean"}
wandb_project = "GNN_citation" #@param {type:"string"}
wandb_run_name = "upload_and_analyze_dataset" #@param {type:"string"}

if use_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name)

dataset_path = 'data/hw3/'
dataset = HW3Dataset(root=dataset_path)
data = dataset[0]


data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes
}

if use_wandb:
    # Log all the details about the data to W&B.
    wandb.log(data_details)
else:
    print(json.dumps(data_details, sort_keys=True, indent=4))

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()
    return fig

if use_wandb:
    # Log the dataset to W&B as an artifact.
    dataset_artifact = wandb.Artifact(name="Citation", type="dataset", metadata=data_details)
    dataset_artifact.add_dir(dataset_path)
    wandb.log_artifact(dataset_artifact)

    # End the W&B run
    wandb.finish()

train_mask = data.train_mask
val_mask = data.val_mask

train_dataset = dataset[:max(train_mask)]
val_dataset = dataset[max(train_mask):]


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# for step, data_ in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data_.num_graphs}')
#     print(data_)
#     print()
# hyperparameters
num_epochs = 2
lr = 0.01

wandb_project = "intro_to_pyg" #@param {type:"string"}
wandb_run_name = "upload_and_analyze_dataset" #@param {type:"string"}

# Initialize W&B run for training
if use_wandb:
    wandb.init(project="GNN_citation")
    wandb.use_artifact("emmyabitbul/GNN_citation/MUTAG:v0")

model = GCN(hidden_channels=64)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Define optimizer.

def train():
    model.train()

    for data_batch in train_loader:  # Iterate in batches over the training dataset.
        out, h = model(data_batch.x, data_batch.edge_index, data_batch.batch)  # Perform a single forward pass.
        loss = criterion(out, data_batch.y.reshape(-1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader, create_table=False):
    model.eval()
    table = wandb.Table(columns=['graph', 'ground truth', 'prediction']) if use_wandb else None
    correct = 0
    loss_ = 0
    for data_batch in loader:  # Iterate in batches over the training/test dataset.
        out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
        loss = criterion(out, data_batch.y)
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        if create_table and use_wandb:
            table.add_data(wandb.Html(plotly.io.to_html(create_graph(data_batch))), data_batch.y.item(), pred.item())

        correct += int((pred == data_batch.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), loss_ / len(loader.dataset), table  # Derive ratio of correct predictions.


for epoch in trange(1, num_epochs):
    train()
    train_acc, train_loss, _ = test(train_loader)
    test_acc, test_loss, test_table = test(val_loader, create_table=True)

    # Log metrics to W&B
    if use_wandb:
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/acc": test_acc,
            "test/loss": test_loss,
            "test/table": test_table
        })

    torch.save(model, "graph_classification_model.pt")

    # Log model checkpoint as an artifact to W&B
    if use_wandb:
        artifact = wandb.Artifact(name="graph_classification_model", type="model")
        artifact.add_file("graph_classification_model.pt")
        wandb.log_artifact(artifact)

# Finish the W&B run
if use_wandb:
    wandb.finish()