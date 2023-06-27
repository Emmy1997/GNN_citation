from dataset import HW3Dataset
from models import GAT, GCN
import torch
import pandas as pd


def predict():
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    model = GAT(num_layers=1, num_heads=3, p=0.4, num_features=128, num_classes=40)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    df = pd.DataFrame({"idx": range(len(preds)), "prediction": preds})
    csv_file = "prediction.csv"
    df.to_csv(csv_file, index=False)


def predict_val():
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    model = GAT(num_layers=1, num_heads=3, p=0.4, num_features=128, num_classes=40)
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    train_mask = data.train_mask
    val_mask = data.val_mask
    out = model(data.x, data.edge_index)
    y_train, y_val = data.y[train_mask].reshape(-1), data.y[val_mask].reshape(-1)  # Perform a single forward pass.
    pred = out.argmax(dim=1)
    pred_train, pred_val = pred[train_mask], pred[val_mask]
    len_val, len_train = len(data.y[val_mask]), len(data.y[train_mask])
    acc_train = int((y_train == pred_train).sum()) / len_train  # Check against ground-truth labels.
    acc_val = int((pred_val == y_val).sum()) / len_val

    df = pd.DataFrame({"idx": range(len(pred_val)), "prediction": pred_val, "true_label": y_val})
    csv_file = "val_prediction.csv"
    df.to_csv(csv_file, index=False)
    print(f"train accuracy: {acc_train}, val accuracy: {acc_val}")


if __name__ == '__main__':
    predict()
