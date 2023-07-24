from typing import Tuple, Dict

import torch
import torch_geometric


def normalize(model: torch.nn.Module, data: torch_geometric.data.Data) -> \
        Tuple[torch.nn.Module, torch_geometric.data.Data]:
    """If the """
    return model, data


def optimize_data(model: torch.nn.Module,
                  dace_models: Dict,
                  data: torch_geometric.data.Data,
                  compute_input_grad: bool) -> Tuple[
    torch.nn.Module, Dict]:
    # Assuming data is in the adjacency list format.
    model, data = normalize(model, data)

    target_formats = {}
    for impl_name, experiment_info in dace_models.items():
        format = experiment_info.graph_format
        format_args = experiment_info.graph_format_args
        format_args_hashable = tuple([v for k, v in sorted(format_args.items())])
        data_idx = (format, format_args_hashable)
        if data_idx not in target_formats:
            target_formats[data_idx] = format.from_pyg_data(data,
                                                            compute_input_grad=compute_input_grad,
                                                            idx_dtype=experiment_info.idx_dtype,
                                                            **format_args)
        experiment_info.data = target_formats[data_idx]
    return model, dace_models
