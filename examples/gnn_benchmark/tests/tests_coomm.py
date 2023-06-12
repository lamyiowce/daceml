import os
from time import sleep

import cupy
import dace
import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from examples.gnn_benchmark.sparse_mm.coomm import coomm

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

    sdfg(A_rows, A_cols, A_vals, B, C)
    if torch.cuda.is_available():
        cupy.cuda.runtime.deviceSynchronize()
    print('Computed: \n', C)
    print('Expected: \n', expected_C)
    assert xp.allclose(C, expected_C)


@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("A_batch_size,B_batch_size", [(3, None), (None, 3), (3, 3)])
def test_batched_spmm(A_batch_size, B_batch_size, transA):
    C_batch_size = A_batch_size or B_batch_size
    N = 2
    M = 4
    F = 5
    A_shape = (N, M) if not transA else (M, N)
    if A_batch_size:
        A_shape = (A_batch_size, *A_shape)
    num_zeros = N*M // 4
    zero_x = torch.randint(low=0, high=A_shape[-2], size=(num_zeros,))
    zero_y = torch.randint(low=0, high=A_shape[-1], size=(num_zeros,))
    B_shape = (B_batch_size, M, F) if B_batch_size else (M, F)
    beta = 0.0
    A = torch.rand(A_shape, dtype=torch.float32)
    A[..., zero_x, zero_y] = 0.0
    B = torch.rand(B_shape, dtype=torch.float32)
    C = torch.empty((C_batch_size, N, F), dtype=torch.float32)
    A_sparse = SparseTensor.from_dense(A.permute(1, 2, 0) if A_batch_size else A)
    A_rows, A_columns, A_vals = A_sparse.coo()
    if A_batch_size:
        A_vals = torch.reshape(A_vals, (A_batch_size, -1))
        A_vals = A_vals.contiguous()
    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.transpose(-1, -2) @ B + beta * C

    @dace.program
    def spmm(A_rows, A_columns, A_vals, B, C):
        coomm(A_rows, A_columns, A_vals, B, C, beta=beta, transA=transA)

    spmm(A_rows, A_columns, A_vals, B, C)

    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)
    sleep(3)


if __name__ == '__main__':
    test_coomm_libnode(beta=0.0, transA=True)
    test_coomm_libnode(beta=1.0, transA=True)
    test_coomm_libnode(beta=0.0, transA=False)
    test_coomm_libnode(beta=1.0, transA=False)
