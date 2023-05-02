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

    # TODO: Not sure if this would work in for data formats with parameters.
    target_formats = {}
    for impl_name, experiment_info in dace_models.items():
        data_fn = experiment_info.convert_data
        if data_fn not in target_formats:
            target_formats[data_fn] = experiment_info.convert_data(data, idx_dtype=experiment_info.idx_dtype)
        experiment_info.data = target_formats[data_fn]
    return model, dace_models
