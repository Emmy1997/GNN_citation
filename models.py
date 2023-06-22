import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features=128, num_classes=40):
        super().__init__()
        torch.manual_seed(1234)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # advanced
        # 2. Readout layer
        # h = global_mean_pool(h)  # [batch_size, hidden_channels]
        #
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)

        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h
