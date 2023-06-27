import wandb
import torch
from tqdm.auto import trange
from models import GCN, GAT
from dataset import HW3Dataset


def training(model, optimizer, criterion, data):
    model.train()
    train_mask = data.train_mask
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[train_mask], data.y[train_mask].reshape(-1))
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    optimizer.zero_grad()  # Clear gradients.
    return loss


def validation(model, criterion, data, create_table=False, use_wandb=False):
    model.eval()
    train_mask = data.train_mask
    val_mask = data.val_mask
    if use_wandb:
        table = wandb.Table(columns=['graph', 'ground truth', 'prediction']) if use_wandb else None
    else:
        table = None
    out = model(data.x, data.edge_index)
    y_train, y_val = data.y[train_mask].reshape(-1), data.y[val_mask].reshape(-1)# Perform a single forward pass.
    val_loss = criterion(out[val_mask], y_val)
    val_loss_ = val_loss.item()
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    pred_train, pred_val = pred[train_mask], pred[val_mask]
    len_val, len_train = len(data.y[val_mask]), len(data.y[train_mask])

    acc_train = int((y_train == pred_train).sum()) / len_train  # Check against ground-truth labels.
    acc_val = int((pred_val == y_val).sum()) / len_val  # Check against ground-truth labels.

    return acc_train, acc_val, val_loss_, table  # Derive ratio of correct predictions.

def train_and_evaluate(use_wandb=False):
    dataset_path = 'data/hw3/'
    dataset = HW3Dataset(root=dataset_path)
    data = dataset[0]
    if use_wandb:
        wandb.init()
    # load_wandb()
        config = wandb.config
        model = GAT(num_layers=config.num_layers, num_heads=config.num_heads, p=config.p)

    else:
        config = {'num_layers': 1, 'num_heads': 3, 'p': 0.4, 'lr': 0.02}
        model = GAT(num_layers=config['num_layers'], num_heads=config['num_heads'], p=config['p'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    if use_wandb:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_val_acc = 0.0
    best_model_path = "best_model.pt"

    for _ in trange(1, 450):
        train_loss = training(model, optimizer, criterion, data)
        train_acc, val_acc, val_loss, val_table = validation(model, criterion, data)

        # Log metrics to W&B
        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/acc": val_acc,
                "val/loss": val_loss,
                "val/table": val_table
            })
        #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), "last_model.pt")  # Save the last model
    print(best_val_acc)
    # Finish the W&B run
    if use_wandb:
        wandb.finish()

train_and_evaluate()