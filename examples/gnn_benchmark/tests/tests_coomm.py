import os
from time import sleep

import cupy
import dace
import numpy as np
import pytest
import torch

from examples.gnn_benchmark.sparse_mm.coomm import coomm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import faulthandler
faulthandler.enable()

@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("transA", [True, False])
def test_coomm_libnode(beta, transA):
    if transA:
        A = cupy.array([[1., 0, 3], [0, 2, 0]]).T
    else:
        A = cupy.array([[1., 0, 3], [0, 2, 0]])
    B = cupy.array([[1., 1], [0, 0], [1, 0]])
    C = cupy.array([[0.1, 0.2], [0.3, 0.4]])
    A_rows, A_cols = cupy.nonzero(A)
    A_rows = cupy.ascontiguousarray(A_rows)
    A_cols = cupy.ascontiguousarray(A_cols)
    A_vals = cupy.ascontiguousarray(A[A_rows, A_cols])
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
    cupy.cuda.runtime.deviceSynchronize()
    print('Computed: \n', C)
    print('Expected: \n', expected_C)
    assert cupy.allclose(C, expected_C)


if __name__ == '__main__':
    test_coomm_libnode(beta=0.0, transA=True)
    test_coomm_libnode(beta=1.0, transA=True)
    test_coomm_libnode(beta=0.0, transA=False)
    test_coomm_libnode(beta=1.0, transA=False)
