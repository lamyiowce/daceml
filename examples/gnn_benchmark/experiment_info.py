import dataclasses
from typing import Callable, Optional

import torch
import torch_geometric

import daceml
from examples.gnn_benchmark import sparse


@dataclasses.dataclass
class ExperimentInfo:
    impl_name: str
    bwd_impl_name: str
    gnn_type: str
    convert_data: Callable[[torch_geometric.data.Data], sparse.GraphMatrix]
    model_eval: daceml.torch.DaceModule
    model_train: daceml.torch.DaceModule
    idx_dtype: torch.dtype
    val_dtype: torch.dtype
    correct: Optional[bool] = None
    correct_grads: Optional[bool] = None
    data: Optional[sparse.GraphMatrix] = None
