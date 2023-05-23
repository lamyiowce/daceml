import numpy as np
import scipy.sparse
import torch

from examples.gnn_benchmark import sparse


def test_hybrid_csr_coo():
    A = torch.tensor([[1, 0, 3, 0, 0, 0, 0], [0, 2, 0, 0, 1, 2, 5], [0, 2., 4.5, 2, 2, 2, 2]])
    A_sparse = scipy.sparse.coo_matrix(A)
    csr_coo_hybrid = sparse.HybridCsrCooGraph(node_features=None,
                                              edge_vals=torch.tensor(A_sparse.data),
                                              rows=torch.Tensor(A_sparse.row).to(torch.int64),
                                              cols=torch.Tensor(A_sparse.col).to(torch.int64),
                                              idx_dtype=torch.int64)

    result_coo = scipy.sparse.coo_matrix(
        (csr_coo_hybrid.coo_edge_vals, (csr_coo_hybrid.coo_rows, csr_coo_hybrid.coo_cols)), shape=A.shape)
    result_csr = scipy.sparse.csr_matrix(
        (csr_coo_hybrid.csr_edge_vals, csr_coo_hybrid.csr_cols, csr_coo_hybrid.csr_rowptrs), shape=A.shape)

    print()
    print(A)
    print(csr_coo_hybrid)
    print(result_csr.toarray() + result_coo.toarray())
    print(result_csr.toarray())
    print(result_coo.toarray())
    assert np.allclose(result_csr.toarray() + result_coo.toarray(), A.numpy())
