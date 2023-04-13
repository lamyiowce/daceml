import dace
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor

from examples.gnn_benchmark import sparse
from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.implementations.common import csrmm_pure
from examples.gnn_benchmark.sparse_mm.blocked_ellpack_mm import blocked_ellpack_mm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("transA", [True, False])
def test_blocked_ellpack_mm_pure(beta, transA):
    A = torch.tensor([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = torch.tensor([[1., 1], [0, 0], [1, 0]])
    C = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    A_ell = sparse.EllpackGraph.from_dense(A, node_features=None)
    _, A_columns, A_values = A_ell.to_input_list()

    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.T @ B + beta * C
    blocked_ellpack_mm(A_ellcolind=A_columns,
                       A_ellvalues=A_values, ellBlockSize=1, B=B, C=C, beta=beta, transA=transA)
    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)


@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("transA", [True, False])
def test_blocked_ellpack_mm_libnode(beta, transA):
    A = torch.tensor([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = torch.tensor([[1., 1], [0, 0], [1, 0]])
    C = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    A_ell = sparse.EllpackGraph.from_dense(A, node_features=None)
    _, A_columns, A_values = A_ell.to_input_list()
    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.T @ B + beta * C

    @dace.program
    def spmm(A_columns, A_vals, B, C):
        blocked_ellpack_mm(A_ellcolind=A_columns,
                           A_ellvalues=A_vals,
                           ellBlockSize=1,
                           B=B,
                           C=C,
                           beta=beta,
                           transA=transA)

    spmm(A_columns, A_values, B, C)

    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)


if __name__ == '__main__':
    test_blocked_ellpack_mm_pure()
    test_blocked_ellpack_mm_libnode(beta=0.0, transA=True)
    test_blocked_ellpack_mm_libnode(beta=1.0, transA=True)
    test_blocked_ellpack_mm_libnode(beta=0.0, transA=False)
    test_blocked_ellpack_mm_libnode(beta=1.0, transA=False)
