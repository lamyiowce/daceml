from typing import Tuple, Dict

import torch
import torch_geometric


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

    # TODO: This converts data for every model, we should only do it once per
    #  type.
    for impl_name, experiment_info in dace_models.items():
        data_in_target_format = experiment_info.data_format.from_pyg_data(data, idx_dtype=experiment_info.idx_dtype)
        experiment_info.data = data_in_target_format
    return model, dace_models
