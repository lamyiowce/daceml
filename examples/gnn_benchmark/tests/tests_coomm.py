import os
from time import sleep

import cupy
import dace
import numpy as np
import pytest
import scipy
import torch
from torch_sparse import SparseTensor

from examples.gnn_benchmark.sparse_mm.coomm import coomm
from examples.gnn_benchmark.tests.common import check_equal

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import faulthandler

faulthandler.enable()


@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("transA", [True, False])
def test_coomm_libnode(beta, transA):
    if torch.cuda.is_available():
        import cupy as xp
    else:
        import numpy as xp

    if transA:
        A = xp.array([[1., 0, 3], [0, 2, 0]]).T
    else:
        A = xp.array([[1., 0, 3], [0, 2, 0]])
    B = xp.array([[1., 1], [0, 0], [1, 0]])
    C = xp.array([[0.1, 0.2], [0.3, 0.4]])
    A_rows, A_cols = xp.nonzero(A)
    A_rows = xp.ascontiguousarray(A_rows)
    A_cols = xp.ascontiguousarray(A_cols)
    A_vals = xp.ascontiguousarray(A[A_rows, A_cols])
    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.T @ B + beta * C

    @dace.program
    def spmm(A_rows, A_columns, A_vals, B, C):
        coomm(A_rows=A_rows,
              A_cols=A_columns,
              A_vals=A_vals,
              B=B,
              C=C,
              beta=beta,
              transA=transA)

    sdfg = spmm.to_sdfg(A_rows, A_cols, A_vals, B, C)
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        sdfg.apply_gpu_transformations()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sdfg(A_rows, A_cols, A_vals, B, C)
    if torch.cuda.is_available():
        cupy.cuda.runtime.deviceSynchronize()
    print('Computed: \n', C)
    print('Expected: \n', expected_C)
    assert xp.allclose(C, expected_C)


@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("A_batch_size,B_batch_size", [(1, 1), (1, 3), (3, 3)])
# @pytest.mark.parametrize("transA", [True])
# @pytest.mark.parametrize("A_batch_size,B_batch_size", [(None, 3)])
def test_batched_spmm(A_batch_size, B_batch_size, transA):
    if torch.cuda.is_available():
        import cupy as xp
        import cupyx.scipy.sparse as xps
    else:
        import numpy as xp
        import scipy.sparse as xps

    xp.random.seed(23)
    np.random.seed(34)

    C_batch_size = max(A_batch_size, B_batch_size)
    N = 2
    M = 4
    F = 5
    A_shape = (N, M) if not transA else (M, N)
    B_shape = (B_batch_size, M, F) if B_batch_size != 1 else (M, F)
    beta = 0.0
    A = xps.random(*A_shape, density=0.5, format='coo')
    B = xp.random.randint(low=-2, high=2, size=B_shape).astype(xp.float32)
    C = xp.zeros((C_batch_size, N, F), dtype=xp.float32)

    A_rows, A_columns = A.row, A.col
    A_rows = xp.ascontiguousarray(A_rows)
    A_columns = xp.ascontiguousarray(A_columns)
    A_batch_size = 1 if A_batch_size is None else A_batch_size
    A_vals = xp.random.randint(low=-1, high=2, size=(A_batch_size, A.nnz)).astype(dtype=xp.float32)

    A_dense = xp.zeros((A_batch_size, *A_shape), dtype=xp.float32)
    for i in range(A_batch_size):
        A_dense[i, A_rows, A_columns] = A_vals[i]
    A_vals = xp.copy(A_vals.squeeze())

    if not transA:
        expected_C = A_dense @ B + beta * C
    else:
        expected_C = xp.transpose(A_dense, (0, -1, -2)) @ B + beta * C

    @dace.program
    def spmm(A_rows, A_columns, A_vals, B, C):
        coomm(A_rows, A_columns, A_vals, B, C, beta=beta, transA=transA)

    sdfg = spmm.to_sdfg(A_rows, A_columns, A_vals, B, C)
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        sdfg.apply_gpu_transformations()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sdfg(A_rows, A_columns, A_vals, B, C)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sleep(3)
    check_equal(expected_pred=expected_C, pred=C)


if __name__ == '__main__':
    test_batched_spmm(A_batch_size=1, B_batch_size=3, transA=True)
    test_batched_spmm(A_batch_size=1, B_batch_size=3, transA=False)
    test_batched_spmm(A_batch_size=3, B_batch_size=3, transA=False)
    test_batched_spmm(A_batch_size=3, B_batch_size=3, transA=True)
    # test_batched_spmm(A_batch_size=3, B_batch_size=3, transA=True)
    # test_coomm_libnode(beta=0.0, transA=True)
    # test_coomm_libnode(beta=1.0, transA=True)
    # test_coomm_libnode(beta=0.0, transA=False)
    # test_coomm_libnode(beta=1.0, transA=False)
