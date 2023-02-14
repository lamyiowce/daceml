from typing import Tuple, Optional

import torch
import torch_geometric

from examples.gnn_benchmark.sparse import CsrGraph, GraphMatrix


def normalize(model: torch.nn.Module, data: torch_geometric.data.Data) -> \
        Tuple[torch.nn.Module, torch_geometric.data.Data]:
    """If the """
    return model, data


def optimize_data(model: torch.nn.Module,
                  data: torch_geometric.data.Data,
                  target_format: Optional[str] = None) -> Tuple[
    torch.nn.Module, GraphMatrix]:
    # Assuming data is in the adjacency list format.
    model, data = normalize(model, data)

    format_converters = {
        "csr": CsrGraph.from_pyg_data,
        "adjacency_list": lambda x: x,
    }

    data_target_format = format_converters[target_format](data)
    return model, data_target_format
