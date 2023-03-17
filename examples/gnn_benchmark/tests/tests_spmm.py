import dace
import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.implementations.common import csrmm_pure

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_implementation(dace_module, implementation):
    def hook(dace_module):
        for node, _ in dace_module.sdfg.all_nodes_recursive():
            if (isinstance(node, dace.sdfg.nodes.LibraryNode)
                    and implementation in node.implementations):
                node.implementation = implementation

    dace_module.prepend_post_onnx_hook("set_implementation", hook)

@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_spmm_libnode(beta):
    A = torch.tensor([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = torch.tensor([[1., 1], [0, 0], [1, 0]])
    C = np.zeros((3, 2)) if beta == 0.0 else np.random.rand(3, 2)
    A_sparse = SparseTensor.from_dense(A)
    A_rowptrs, A_columns, A_vals = A_sparse.csr()
    expected_C = A @ B + beta * C


    @dace.program
    def spmm(A_rowptrs, A_columns, A_vals, B, C):
        csrmm(A_rowptrs, A_columns, A_vals, B, C, beta=beta)

    spmm(A_rowptrs, A_columns, A_vals, B, C)

    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)

def test_spmm_pure():
    A = torch.tensor([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = torch.tensor([[1., 1], [0, 0], [1, 0]])
    C = np.zeros((3, 2))
    A_sparse = SparseTensor.from_dense(A)
    A_rowptrs, A_columns, A_values = A_sparse.csr()

    expected_C = A @ B
    csrmm_pure(A_rowptrs, A_columns, A_values, B, C, N=3, K=2, beta=0.0)
    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)
