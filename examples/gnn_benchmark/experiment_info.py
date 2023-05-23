import dataclasses
from typing import Callable, Optional, Any, Dict, Type

import torch
import torch_geometric

from examples.gnn_benchmark import sparse


@dataclasses.dataclass
class ExperimentInfo:
    impl_name: str
    bwd_impl_name: str
    gnn_type: str
    graph_format: Type[sparse.GraphMatrix]
    graph_format_args: Dict[str, Any]
    model_eval: 'daceml.torch.DaceModule'
    model_train: 'daceml.torch.DaceModule'
    idx_dtype: torch.dtype
    val_dtype: torch.dtype
    correct: Optional[bool] = None
    correct_grads: Optional[bool] = None
    data: Optional[sparse.GraphMatrix] = None
