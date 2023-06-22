import torch
from torch.nn import Linear, ReLU, Tanh
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import global_mean_pool
# import torch.nn.functional as F


class simple_GCN(torch.nn.Module):
    def __init__(self, activation, hidden_channels, num_features=128, num_classes=40):
        super().__init__()
        torch.manual_seed(1234)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, num_classes)
        self.activation = ReLU() if activation == 'relu' else Tanh

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = self.conv2(h, edge_index)
        h = self.activation(h)
        h = self.conv3(h, edge_index)
        h = self.activation(h)

        # advanced
        # 2. Readout layer
        # h = global_mean_pool(h)  # [batch_size, hidden_channels]
        #
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)

        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h


class GCN(torch.nn.Module):
    def __init__(self, activation, hidden_channels, num_layers=3, num_features=128, num_classes=40):
        super().__init__()
        torch.manual_seed(1234)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.activations.append(ReLU() if activation == 'relu' else Tanh())

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.activations[i](h)
        out = self.classifier(h)
        return out, h
