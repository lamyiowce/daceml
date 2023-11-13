import os
from typing import Literal

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.nn
import dgl.data
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

GCNCONV_IMPLEMENTATIONS = {
    "pyg": GCNConv,
    "dgl": dgl.nn.GraphConv,
}

GCNCONV_ARGS = {
    "pyg": {"normalize": False, "add_self_loops": False},
    "dgl": {"norm": "none", "allow_zero_in_degree": False},
}


class GCNSingleLayer(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes, num_layers,
                 compute_input_grad=False,
                 bias=True, bias_init=torch.nn.init.zeros_,
                 implementation: Literal["pyg", "dgl"] = "pyg"):
        del num_classes
        del num_layers
        super().__init__()

        layer = GCNCONV_IMPLEMENTATIONS[implementation]
        layer_args = GCNCONV_ARGS[implementation]
        self.conv = layer(num_node_features,
                          num_hidden_features,
                          **layer_args,
                          bias=bias)
        if bias:
            bias_init(self.conv.bias)
        self.conv.is_first = not compute_input_grad

    def forward(self, x, *edge_info):
        x = self.conv(x, *edge_info)

        return x


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features, num_classes, num_layers,
                 compute_input_grad=False,
                 bias=True, bias_init=torch.nn.init.zeros_, implementation="pyg"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.implementation = implementation
        layer = GCNCONV_IMPLEMENTATIONS[implementation]
        layer_args = GCNCONV_ARGS[implementation]
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else num_hidden_features
            out_channels = num_classes if i == num_layers - 1 else num_hidden_features
            conv = layer(in_channels,
                         out_channels,
                         **layer_args,
                         bias=bias)
            if bias:
                bias_init(conv.bias)
            if compute_input_grad:
                conv.is_first = False
            else:
                conv.is_first = i == 0
            self.convs.append(conv)

        self.act = nn.ReLU()

    def forward(self, x, *edge_info):
        for conv in self.convs[:-1]:
            x = self.call_layer(conv, x, *edge_info)  # Size: `hidden size`
            x = self.act(x)  # ReLU
        x = self.call_layer(self.convs[-1], x, *edge_info)  # Size: `num_classes`
        return x

    def call_layer(self, layer, x, *edge_info):
        if self.implementation == "dgl":
            graph = edge_info[0]
            x = layer(graph, x)
        else:
            x = layer(x, *edge_info)
        return x


class LinearModel(torch.nn.Module):
    def __init__(self, num_node_features, _unused, num_classes,
                 _unused2) -> None:
        super().__init__()
        self.lin = nn.Linear(num_node_features, num_classes)

    def forward(self, x):
        x = self.lin(x)
        return x


def make_gat_args(gat_layer, heads: int):
    gat_args = {
        GATConv: {"add_self_loops": False, "heads": heads},
        CuGraphGATConv: {"heads": heads},
        dgl.nn.GATConv: {"allow_zero_in_degree": True, "num_heads": heads},
    }
    return gat_args[gat_layer]


class GAT(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 features_per_head,
                 num_classes,
                 num_layers,
                 gat_layer=GATConv,
                 num_heads=8,
                 bias=True,
                 bias_init=torch.nn.init.zeros_):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(gat_layer(num_node_features,
                                        num_classes,
                                        bias=bias,
                                        **make_gat_args(gat_layer, heads=1)))
        else:
            self.convs.append(gat_layer(num_node_features,
                                        features_per_head,
                                        bias=bias,
                                        **make_gat_args(gat_layer, heads=num_heads)))

            for i in range(num_layers - 2):
                conv = gat_layer(features_per_head * num_heads,
                                 features_per_head,
                                 heads=num_heads,
                                 bias=bias,
                                 **make_gat_args(gat_layer, heads=num_heads))
                self.convs.append(conv)

            self.convs.append(gat_layer(features_per_head * num_heads,
                                        num_classes,
                                        bias=bias,
                                        **make_gat_args(gat_layer, heads=1)))
        if bias:
            for layer in self.convs:
                bias_init(layer.bias)
        self.act = nn.ELU()

    def forward(self, x, *edge_info):
        for conv in self.convs[:-1]:
            x = self.call_layer(conv, x, *edge_info)  # Size: `hidden size`
            x = self.act(x)  # ReLU
        x = self.call_layer(self.convs[-1], x, *edge_info)  # Size: `num_classes`
        return x

    def call_layer(self, layer, x, *edge_info):
        if isinstance(layer, dgl.nn.GATConv):
            graph = edge_info[0]
            x = layer(graph, x)  # num_nodes x num_heads x features_per_head
            x = x.flatten(start_dim=1)  # num_nodes x (num_heads * features_per_head)
        else:
            x = layer(x, *edge_info)
        return x


class GATSingleLayer(torch.nn.Module):
    def __init__(self, num_node_features, features_per_head, num_classes,
                 num_layers,
                 gat_layer=GATConv,
                 num_heads=8, bias=True, bias_init=torch.nn.init.zeros_):
        del num_classes
        del num_layers
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
