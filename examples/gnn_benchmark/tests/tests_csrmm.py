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
@pytest.mark.parametrize("transA", [True, False])
def test_spmm_libnode(beta, transA):
    A = torch.tensor([[1, 0, 3], [0, 2, 0], [0, 2., 4.5]])
    B = torch.tensor([[1., 1], [0, 0], [1, 0]])
    C = torch.zeros((3, 2)) if beta == 0.0 else torch.rand(3, 2)
    A_sparse = SparseTensor.from_dense(A)
    A_rowptrs, A_columns, A_vals = A_sparse.csr()
    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = A.T @ B + beta * C

    @dace.program
    def spmm(A_rowptrs, A_columns, A_vals, B, C):
        csrmm(A_rowptrs, A_columns, A_vals, B, C, beta=beta, transA=transA)

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

@pytest.mark.parametrize("transA", [True, False])
def test_batched_spmm(transA):
    batch_size = 3
    N = 2
    M = 4
    F = 5
    beta = 0.0
    A = torch.rand((batch_size, N, M), dtype=torch.float32)
    zero_x = torch.randint(0, N)
    zero_y = torch.randint(0, M)
    A[:, zero_x, zero_y] = 0.0
    B = torch.rand((batch_size, M, F), dtype=torch.float32)
    C = torch.empty((batch_size, N, F), dtype=torch.float32)
    A_sparse = SparseTensor.from_dense(A)
    A_rowptrs, A_columns, A_vals = A_sparse.csr()

    if not transA:
        expected_C = A @ B + beta * C
    else:
        expected_C = np.transpose(A, (0, 2, 1)) @ B + beta * C

    @dace.program
    def spmm(A_rowptrs, A_columns, A_vals, B, C):
        csrmm(A_rowptrs, A_columns, A_vals, B, C, beta=beta, transA=transA)

    spmm(A_rowptrs, A_columns, A_vals, B, C)

    print('\nCalculated: \n', C)
    print('Expected: \n', expected_C)
    assert np.allclose(C, expected_C)

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

    sdfg = compute_spmms.to_sdfg(A_rowptrs, A_columns, A_values, B, C1, C2, C3, C_sum)
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
