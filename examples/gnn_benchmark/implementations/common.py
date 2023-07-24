import abc
from enum import Enum
from typing import Type, Dict, Union

import dace
import torch

from daceml.onnx import ONNXForward
from examples.gnn_benchmark import sparse


class SpecialInputType(Enum):
    """ Types of special inputs that can be passed to a dace model. """
    IDX_DTYPE = 1  # Uses dtype specified in experiment info.
    VAL_DTYPE = 2  # Uses dtype specified in experiment info.


class SparseLayerBase(ONNXForward, metaclass=abc.ABCMeta):

    @staticmethod
    @property
    @abc.abstractmethod
    def graph_format() -> Type[sparse.GraphMatrix]:
        raise NotImplementedError

    @staticmethod
    @property
    @abc.abstractmethod
    def input_spec() -> Dict[str, Union[dace.typeclass, SpecialInputType]]:
        raise NotImplementedError

    output_spec = {"output": SpecialInputType.VAL_DTYPE}
    buffer_spec = []



    allowed_idx_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]


@dace.program
def csrmm_pure(A_rowptrs,
               A_columns,
               A_values,
               B,
               C,
               N,
               K,
               beta=0.
               ):
    if beta != 0. and beta != 1.:
        C[:] = beta * C
    elif beta == 0:
        C[:] = 0
    for i, k in dace.map[0:N, 0:K]:
        for j in dace.map[A_rowptrs[i]:A_rowptrs[i + 1]]:
            column = A_columns[j]
            mult = B[column, k] * A_values[j]
            C[i, k] += mult
