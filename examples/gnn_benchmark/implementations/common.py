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
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
        raise NotImplementedError
