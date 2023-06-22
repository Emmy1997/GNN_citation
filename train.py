import json
import plotly
import wandb
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from tqdm.auto import trange
from visualize import GraphVisualization
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



# for step, data_ in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data_.num_graphs}')
#     print(data_)
#     print()
# hyperparameters
num_epochs = 10
lr = 0.01


# Initialize W&B run for training
if use_wandb:
    wandb.init(project="GNN_citation")
    wandb.use_artifact("emmyabitbul/GNN_citation/MUTAG:v0")

model = GCN(hidden_channels=64)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Define optimizer.

def training():
    model.train()

    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[train_mask], data.y[train_mask].reshape(-1))
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    return loss, h


def validation(create_table=False):
    model.eval()
    table = wandb.Table(columns=['graph', 'ground truth', 'prediction']) if use_wandb else None
    out, h = model(data.x, data.edge_index)
    y_train, y_val = data.y[train_mask].reshape(-1), data.y[val_mask].reshape(-1)# Perform a single forward pass.
    val_loss = criterion(out[val_mask], y_val)
    val_loss_ = val_loss.item()
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    pred_train, pred_val = pred[train_mask], pred[val_mask]
    if create_table and use_wandb:
        table.add_data(wandb.Html(plotly.io.to_html(create_graph(data))), data.y.item(), pred.item())
    len_val, len_train = len(data.y[val_mask]), len(data.y[train_mask])

    acc_train = int((y_train == pred_train).sum()) / len_train  # Check against ground-truth labels.
    acc_val = int((pred_val == y_val).sum()) / len_val  # Check against ground-truth labels.

    return acc_train, acc_val, val_loss_, table  # Derive ratio of correct predictions.


for epoch in trange(1, num_epochs):
    train_loss, h = training()
    train_acc, val_acc, val_loss, val_table = validation()


    # Log metrics to W&B
    if use_wandb:
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/acc": val_acc,
            "val/loss": val_loss,
            "val/table": val_table
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