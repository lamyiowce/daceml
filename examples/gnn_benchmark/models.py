import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


class GCNSingleLayer(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes,
                 normalize, bias_init=torch.nn.init.zeros_):
        del num_classes
        del normalize
        super().__init__()
        self.conv = GCNConv(num_node_features,
                            num_hidden_features,
                            normalize=False,
                            add_self_loops=False)
        bias_init(self.conv.bias)

    def forward(self, x, *edge_info):
        x = self.conv(x, *edge_info)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes,
                 normalize, bias_init=torch.nn.init.zeros_):
        super().__init__()
        self.conv1 = GCNConv(num_node_features,
                             num_hidden_features,
                             normalize=normalize,
                             add_self_loops=False)
        self.conv2 = GCNConv(num_hidden_features,
                             num_classes,
                             normalize=normalize,
                             add_self_loops=False)
        bias_init(self.conv1.bias)
        bias_init(self.conv2.bias)

        self.act = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)  # Size: `hidden size`
        x = self.act(x)  # ReLU
        x = self.conv2(x, *edge_info)  # Size: `num_classes`

        return self.log_softmax(x)


class LinearModel(torch.nn.Module):
    def __init__(self, num_node_features, _unused, num_classes,
                 _unused2) -> None:
        super().__init__()
        self.lin = nn.Linear(num_node_features, num_classes)

    def forward(self, x):
        x = self.lin(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 features_per_head,
                 num_classes,
                 _unsued,
                 num_heads=8,
                 bias_init=torch.nn.init.zeros_):
        super().__init__()
        self.conv1 = GATConv(num_node_features,
                             features_per_head,
                             heads=num_heads,
                             add_self_loops=False,
                             bias=True)
        self.conv2 = GATConv(features_per_head * num_heads,
                             num_classes,
                             heads=1,
                             add_self_loops=False,
                             bias=True)

        self.act = nn.ELU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)
        x = self.act(x)
        x = self.conv2(x, *edge_info)

        return self.log_softmax(x)
