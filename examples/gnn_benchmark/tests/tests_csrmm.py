import os
from time import sleep

import dace
import numpy as np
import pytest
import scipy
import torch
from torch_sparse import SparseTensor

from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.implementations.common import csrmm_pure
from examples.gnn_benchmark.tests.common import check_equal

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
@pytest.mark.parametrize("transA", [True, False])
def test_spmm_libnode(beta, transA):
    if torch.cuda.is_available():
        import cupy as xp
        import cupyx.scipy.sparse as xps
    else:
        import numpy as xp
        import scipy.sparse as xps

    A = xp.array([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = xp.array([[1., 1], [0, 0], [1, 0]])
    C = xp.random.rand(3, 2)

    A_sparse = xps.csr_matrix(A)

    A_rowptrs = xp.array(A_sparse.indptr)
    A_columns = xp.array(A_sparse.indices)
    A_vals = xp.array(A_sparse.data)

    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.T @ B + beta * C

    @dace.program
    def spmm(A_rowptrs, A_columns, A_vals, B, C):
        csrmm(A_rowptrs, A_columns, A_vals, B, C, beta=beta, transA=transA)

    sdfg = spmm.to_sdfg(A_rowptrs, A_columns, A_vals, B, C)

    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        sdfg.apply_gpu_transformations()

    sdfg(A_rowptrs, A_columns, A_vals, B, C)

    sleep(1)
    check_equal(expected_pred=expected_C, pred=C)


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


@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("A_batch_size,B_batch_size",
                         [(None, None), (None, 3), (3, 3)])
@pytest.mark.parametrize("N,M,F", [(3, 4, 2), (40, 30, 20)])
@pytest.mark.parametrize("beta", [0.0, 1.0])
# @pytest.mark.parametrize("transA", [True])
# @pytest.mark.parametrize("A_batch_size,B_batch_size", [(None, 3)])
def test_batched_spmm(A_batch_size, B_batch_size, transA, N, M, F, beta):
    if torch.cuda.is_available():
        import cupy as xp
        import cupyx.scipy.sparse as xps
    else:
        import numpy as xp
        import scipy.sparse as xps
    xp.random.seed(23)
    np.random.seed(34)

    C_batch_size = A_batch_size or B_batch_size or 1
    A_shape = (N, M) if not transA else (M, N)
    B_shape = (B_batch_size, M, F) if B_batch_size else (M, F)
    A = xps.random(*A_shape, density=0.5, format='csr')
    B = xp.random.randint(low=-2, high=2, size=B_shape).astype(xp.float32)
    C = xp.random.randint(low=-2, high=3, size=(C_batch_size, N, F)).astype(xp.float32) * 10

    A_rowptrs, A_columns = A.indptr, A.indices
    A_rowptrs = xp.copy(xp.asarray(A_rowptrs))
    A_columns = xp.copy(xp.asarray(A_columns))
    A_columns = xp.copy(xp.concatenate([A_columns] * A_batch_size, axis=0))
    A_batch_size = 1 if A_batch_size is None else A_batch_size
    A_vals = xp.random.randint(low=-1, high=2,
                               size=(A_batch_size, A.nnz)).astype(
        dtype=xp.float32)

    print("NNZ: ", A.nnz)
    A_dense = xp.zeros((A_batch_size, *A_shape), dtype=xp.float32)
    for i in range(A_batch_size):
        A.data[:] = A_vals[i]
        A_dense[i, :, :] = A.todense()
    A_vals = A_vals.squeeze()

    print("A: ", A_dense)
    print("B: ", B)
    if not transA:
        expected_C = A_dense @ B + beta * C
    else:
        expected_C = xp.transpose(A_dense, (0, -1, -2)) @ B + beta * C

    @dace.program
    def spmm(A_rowptrs, A_columns, A_vals, B, C):
        csrmm(A_rowptrs, A_columns, A_vals, B, C, beta=beta, transA=transA)

    sdfg = spmm.to_sdfg(A_rowptrs, A_columns, A_vals, B, C)
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        sdfg.apply_gpu_transformations()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sdfg(A_rowptrs, A_columns, A_vals, B, C)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sleep(3)
    check_equal(expected_pred=expected_C, pred=C)


def test_many_stream_spmm():
    N = 4
    M = 2
    device = 'cuda'
    B = torch.rand((N, M), dtype=torch.float32, device=device)
    A = torch.tensor([[1., 0, 1, 0],
                      [1., 1, 1, 0],
                      [0., 1, 1, 1],
                      [0., 0, 1, 0]], device=device, dtype=torch.float32)
    A_sparse = SparseTensor.from_dense(A).to(device)
    A_rowptrs, A_columns, A_values = A_sparse.csr()

    C1 = torch.zeros((N, M), dtype=torch.float32).to(device)
    C2 = torch.zeros((N, M), dtype=torch.float32).to(device)
    C3 = torch.zeros((N, M), dtype=torch.float32).to(device)
    C_sum = torch.zeros((N, M), dtype=torch.float32).to(device)

    @dace.program
    def compute_spmms(A_rowptrs, A_cols, A_vals, B, C1, C2, C3, C_sum):
        # runs three SPMMS.
        # C1 = A * B
        csrmm(A_rowptrs, A_cols, A_vals, B, C1, transA=False)
        csrmm(A_rowptrs, A_cols, A_vals, B, C2, transA=False)
        csrmm(A_rowptrs, A_cols, A_vals, B, C3, transA=False)
        C_sum[:] = C1 + C2 + C3

    sdfg = compute_spmms.to_sdfg(A_rowptrs, A_columns, A_values, B, C1, C2, C3,
                                 C_sum)
    sdfg.apply_gpu_transformations()
    sdfg(A_rowptrs, A_columns, A_values, B, C1, C2, C3, C_sum)

    expected_single = (A @ B).cpu().numpy()
    print("Calculated C1: ", C1)
    print("Calculated C2: ", C2)
    print("Calculated C3: ", C3)
    print("Expected: ", expected_single)
    print("Calculated Csum: ", C_sum)
    print("Expected: ", 3 * expected_single)
    assert np.allclose(C2.cpu().numpy(), expected_single)
    assert np.allclose(C3.cpu().numpy(), expected_single)
    assert np.allclose(C1.cpu().numpy(), expected_single)
    assert np.allclose(C_sum.cpu().numpy(), 3 * expected_single)


if __name__ == '__main__':
    test_batched_spmm(A_batch_size=None, B_batch_size=None, transA=False)
    test_batched_spmm(A_batch_size=None, B_batch_size=None, transA=True)
    test_batched_spmm(A_batch_size=None, B_batch_size=2, transA=False)
    test_batched_spmm(A_batch_size=None, B_batch_size=2, transA=True)
    test_batched_spmm(A_batch_size=2, B_batch_size=2, transA=False)
    test_batched_spmm(A_batch_size=2, B_batch_size=2, transA=True)
