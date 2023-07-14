import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv

try:
    from torch_geometric.nn import CuGraphGATConv
except ImportError:
    CuGraphGATConv = None

try:
    from torch_geometric.nn import FusedGATConv
except ImportError:
    FusedGATConv = None


class GCNSingleLayer(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes,
                 bias=True, bias_init=torch.nn.init.zeros_):
        del num_classes
        super().__init__()
        self.conv = GCNConv(num_node_features,
                            num_hidden_features,
                            normalize=False,
                            add_self_loops=False,
                            bias=bias)
        if bias:
            bias_init(self.conv.bias)

    def forward(self, x, *edge_info):
        x = self.conv(x, *edge_info)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes,
                 bias=True, bias_init=torch.nn.init.zeros_):
        super().__init__()
        self.conv1 = GCNConv(num_node_features,
                             num_hidden_features,
                             normalize=False,
                             add_self_loops=False,
                             bias=bias)
        self.conv2 = GCNConv(num_hidden_features,
                             num_classes,
                             normalize=False,
                             add_self_loops=False,
                             bias=bias)
        if bias:
            bias_init(self.conv1.bias)
            bias_init(self.conv2.bias)

        self.act = nn.ReLU()

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)  # Size: `hidden size`
        x = self.act(x)  # ReLU
        x = self.conv2(x, *edge_info)  # Size: `num_classes`
        return x


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
                 gat_layer=GATConv,
                 num_heads=8,
                 bias=True,
                 bias_init=torch.nn.init.zeros_):
        super().__init__()
        if gat_layer == CuGraphGATConv:
            additional_kwargs = {}
        else:
            additional_kwargs = {"add_self_loops": False}
        self.conv1 = gat_layer(num_node_features,
                               features_per_head,
                               heads=num_heads,
                               bias=bias,
                               **additional_kwargs)
        self.conv2 = gat_layer(features_per_head * num_heads,
                               num_classes,
                               heads=1,
                               bias=bias,
                               **additional_kwargs)
        if bias:
            bias_init(self.conv1.bias)
            bias_init(self.conv2.bias)
        self.act = nn.ELU()

    def forward(self, x, *edge_info):
        x = self.conv1(x, *edge_info)
        x = self.act(x)
        x = self.conv2(x, *edge_info)

        return x


class GATSingleLayer(torch.nn.Module):
    def __init__(self, num_node_features, features_per_head, num_classes,
                 gat_layer=GATConv,
                 num_heads=8, bias=True, bias_init=torch.nn.init.zeros_):
        del num_classes
        super().__init__()
        if gat_layer == CuGraphGATConv:
            additional_kwargs = {}
        else:
            additional_kwargs = {"add_self_loops": False}
        self.conv = gat_layer(num_node_features,
                              features_per_head,
                              heads=num_heads,
                              bias=bias,
                              **additional_kwargs)
        if bias:
            bias_init(self.conv.bias)

    def forward(self, x, *edge_info):
        x = self.conv(x, *edge_info)

        return x
