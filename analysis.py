from utils import visualize_graph, visualize_embedding
from dataset import HW3Dataset
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_confusion_matrix(df = None):
    y_true = df['true_label']
    y_pred = df['prediction']
    confusion = confusion_matrix(y_true, y_pred)
    # Create a DataFrame from the confusion matrix
    class_labels = range(40)  # 40 classes

    confusion_df = pd.DataFrame(confusion, index=class_labels, columns=class_labels)

    # Create a heatmap using seaborn
    plt.figure(figsize=(22, 15))
    sns.heatmap(confusion_df, annot=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig('confusion_mat.png')

#
if __name__ == '__main__':
    df = pd.read_csv('val_prediction.csv')
    create_confusion_matrix(df)

#     dataset = HW3Dataset(root='data/hw3/')
#     data = dataset[0]
#     print(data)
#     # Gather some statistics about the graph.
#     print(f'Number of nodes: {data.num_nodes}')
#     print(f'Number of edges: {data.num_edges}')
#     print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#     print(f'Number of training nodes: {data.train_mask.shape[0]}')
#     print(f'Training node label rate: {int(data.train_mask.shape[0]) / data.num_nodes:.2f}')
#     print(f'Has isolated nodes: {data.has_isolated_nodes()}')
#     print(f'Has self-loops: {data.has_self_loops()}')
#     print(f'Is undirected: {data.is_undirected()}')
#
#     G = to_networkx(data, to_undirected=True)
#     visualize_graph(G, color=data.y)

