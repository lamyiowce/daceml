from typing import Tuple, Dict

import torch
import torch_geometric

from examples.gnn_benchmark import sparse


def normalize(model: torch.nn.Module, data: torch_geometric.data.Data) -> \
        Tuple[torch.nn.Module, torch_geometric.data.Data]:
    """If the """
    return model, data


def optimize_data(model: torch.nn.Module,
                  dace_models: Dict,
                  data: torch_geometric.data.Data) -> Tuple[
    torch.nn.Module, Dict]:
    # Assuming data is in the adjacency list format.
    model, data = normalize(model, data)

    format_converters = {
        "csr": sparse.CsrGraph.from_pyg_data,
        "coo": sparse.CooGraph.from_pyg_data,
        "csc": sparse.CscGraph.from_pyg_data,
        "ellpack": sparse.EllpackGraph.from_pyg_data,
        "ellpack_t": sparse.EllpackTransposedGraph.from_pyg_data,
        "adjacency_list": lambda x: x,
    }

    for impl_name, model_info in dace_models.items():
        data_in_target_format = model_info.data_format.from_pyg_data(data)
        model_info.data = data_in_target_format
    return model, dace_models
