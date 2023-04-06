import abc
from typing import Type, Dict

import dace

from daceml.onnx import ONNXForward
from examples.gnn_benchmark import sparse


class SparseLayerBase(ONNXForward, metaclass=abc.ABCMeta):

    @staticmethod
    @property
    @abc.abstractmethod
    def graph_format() -> Type[sparse.GraphMatrix]:
        raise NotImplementedError

    @staticmethod
    @property
    @abc.abstractmethod
    def input_spec() -> Dict[str, dace.typeclass]:
        raise NotImplementedError

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int, num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        raise NotImplementedError


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
