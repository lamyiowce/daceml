import copy
from typing import Dict

import numpy as np
import torch
from scipy import sparse

from examples.gnn_benchmark.tests.common import check_equal, check_grads, pred_torch, setup_data
from examples.gnn_benchmark.tests.tests_mygat import MyGat

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def gat_forward(N: int, A: sparse.csr_matrix, H: np.ndarray,
                W, a_l, a_r):
    """ Forward pass of GAT layer on CPU, only for debugging purposes.

        param A: adjacency matrix
        param input: input H matrix
    """
    # M = H @ W, l = M @ a_l, r = M @ a_r
    # M = np.zeros((A_dim, self.out_channel), dtype=H.dtype)
    M = H @ W
    l = M @ a_l
    r = M @ a_r

    # D = l + r^T
    D_data = np.zeros_like(A.data, H.dtype)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            D_data[j] = l[i] + r[A.indices[j]]

    # leaky relu
    E_data = np.maximum(D_data, 0.2 * D_data)

    # softmax row-wise
    row_max = np.full(N, np.NINF, dtype=H.dtype)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            row_max[i] = max(row_max[i], E_data[j])

    row_sum = np.zeros(N, dtype=H.dtype)
    Alpha_data = E_data  # shallow copy
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            # exp(x - max(x))
            Alpha_data[j] = np.exp(Alpha_data[j] - row_max[i])
            row_sum[i] += Alpha_data[j]
    eps = np.finfo(row_sum.dtype).eps
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            Alpha_data[j] /= (row_sum[i])

    # Z = Alpha @ M
    # Z = np.zeros_like(M, dtype=H.dtype)
    # for i in range(len(A.indptr) - 1):
    #     for j in range(A.indptr[i], A.indptr[i + 1]):
    #         Z[i, :] += M[A.indices[j], :] * Alpha_data[j]

    sparse_Alpha = sparse.csr_matrix((Alpha_data, A.indices, A.indptr),
                                     shape=A.shape)
    Z = sparse_Alpha @ M

    # # relu
    # output = np.maximum(Z, 0)

    # cache data if needed in backward pass
    ctx = {}
    ctx['H'] = H
    ctx['M'] = M
    ctx['l'] = l
    ctx['r'] = r
    # ctx['D_data'] = D_data
    ctx['Alpha_data'] = Alpha_data
    ctx['Z'] = Z
    return Z, ctx


def gat_backward(N: int, ctx: Dict[str, np.array], A: sparse.csr_matrix,
                 grad_out: np.ndarray, a_l, a_r, W):
    """ Backward pass of GAT layer on CPU, only for debugging purposes.

        param A: adjacency matrix
        param grad_out: gradient of output
    """
    A_dim = N
    # relu
    # dZ = grad_out * (ctx['Z'] > 0)
    dZ = grad_out

    # dAlpha = dZ * M^T
    d_Alpha_data = np.zeros_like(A.data, dtype=grad_out.dtype)
    for row in range(len(A.indptr) - 1):
        for col_idx in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[col_idx]
            d_Alpha_data[col_idx] = np.dot(dZ[row, :], ctx['M'][col, :])

    # dE[i,:] = (dAlpha[i,:] - dAlpha[i,:] @ Alpha[i,:].T) * Alpha[i,:]
    dot_prod = np.zeros(A.shape[0], dtype=grad_out.dtype)
    for i in range(len(A.indptr) - 1):
        dot_prod[i] = np.dot(
            ctx['Alpha_data'][A.indptr[i]:A.indptr[i + 1]],
            d_Alpha_data[A.indptr[i]:A.indptr[i + 1]])

    dE_data = np.zeros_like(A.data, dtype=grad_out.dtype)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            dE_data[j] = (d_Alpha_data[j] - dot_prod[i]) * \
                         ctx['Alpha_data'][j]

    # leaky relu
    dD_data = np.zeros_like(A.data, dtype=grad_out.dtype)
    D_data = np.zeros_like(A.data, dtype=grad_out.dtype)
    for i in range(len(A.indptr) - 1):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            D_data[j] = ctx['l'][i] + ctx['r'][A.indices[j]]
    dD_data = dE_data * (D_data > 0) + 0.2 * dE_data * (D_data <= 0)

    # dl = sum_row dD, dr = sum_col dD
    dl = np.zeros(A_dim, dtype=grad_out.dtype)
    dr = np.zeros(A_dim, dtype=grad_out.dtype)
    for row in range(len(A.indptr) - 1):
        for col_idx in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[col_idx]
            dl[row] += dD_data[col_idx]
            dr[col] += dD_data[col_idx]

    # dM = dl @ a_l.T + dr @ a_r.T + Alpha.T @ dZ
    dM = np.zeros_like(ctx['M'], dtype=grad_out.dtype)
    dM[0:N, :] += dl[0:N, None] @ a_l[None, :]
    dM[0:N, :] += dr[0:N, None] @ a_r[None, :]
    Alpha = sparse.csr_matrix((ctx['Alpha_data'], A.indices, A.indptr),
                              shape=A.shape)
    dM[0:N, :] += Alpha.T @ dZ[0:N, :]

    # da_l = M^T @ dl, da_r = M^T @ dr
    da_l = ctx['M'].T @ dl
    da_r = ctx['M'].T @ dr

    # dH = dM @ W^T
    dH = np.zeros_like(ctx['H'], dtype=grad_out.dtype)
    dH[0:N, :] = dM[0:N, :] @ W.T

    # dW = H^T @ dM
    dW = ctx['H'][0:A.shape[0], :].T @ dM[0:A.shape[0], :]

    grads = {
        'lin_src.weight': dW.T,
        'x': dH,
        'att_src': da_r,
        'att_dst': da_l,
        'H_prime': dM,
        'att_weights': sparse.csr_matrix((d_Alpha_data, A.indices, A.indptr),
                                         shape=A.shape).todense(),
    }

    return grads


def test_paper():
    N = 3
    F_in = 2
    F_out = 4
    heads = 1

    adj_matrix, layer, random_mask, x = setup_data(N, F_in, F_out, heads)
    adj_matrix_dense = adj_matrix.to_dense()

    att_weights, pred = pred_torch(adj_matrix, layer, random_mask, x)

    # Use a "proxy" implementation to be able to check the inner gradients as well.
    mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
                  att_dst=layer.att_dst)
    mygat_x = copy.deepcopy(x)
    mygat_x.grad = None
    mygat_out = mygat.forward(mygat_x, adj_matrix_dense)
    assert torch.allclose(mygat_out, pred)
    assert check_equal(mygat.att_weights.detach().numpy()[..., None],
                       att_weights.to_dense().detach().numpy())
    loss = torch.sum(mygat_out * random_mask)
    mygat.H_prime.retain_grad()
    mygat.att_weights.retain_grad()
    loss.backward()

    with torch.no_grad():
        paper_fwd_result, ctx = gat_forward(x.shape[0],
                                            sparse.csr_matrix(
                                                adj_matrix_dense.T),
                                            H=x.numpy(),
                                            W=layer.lin_src.weight.numpy().T,
                                            a_r=layer.att_src[0, 0].numpy(),
                                            a_l=layer.att_dst[0, 0].numpy())
        assert torch.allclose(torch.tensor(paper_fwd_result), pred)
        paper_grads = gat_backward(N=N, ctx=ctx,
                                   A=sparse.csr_matrix(adj_matrix_dense.T),
                                   grad_out=pred.grad.numpy(),
                                   a_r=layer.att_src[0, 0].numpy(),
                                   a_l=layer.att_dst[0, 0].numpy(),
                                   W=layer.lin_src.weight.numpy().T)

    params = dict(layer.named_parameters())
    params['x'] = x
    params['H_prime'] = mygat.H_prime
    params['att_weights'] = mygat.att_weights

    messages = []
    for name, param in params.items():
        # print(name, param.grad)
        if paper_grads[name] is not None:
            correct, message = check_equal(param.grad.numpy(),
                                           paper_grads[name],
                                           name=name + ' grad', do_assert=False)
            if not correct:
                messages.append(message)
                print(message)

    assert len(messages) == 0, str(messages)

    check_grads(expected_params=params, result=paper_grads)


def compute_grads_csc(x, A_rowptrs, A_cols, adj_matrix, weight,
                      att_src, att_dst,
                      bias,
                      att_weights,
                      output,
                      out_grad):
    # x: N x F_in
    # adj_matrix: N x N
    # weight: F_out x F_in
    # att_src: 1 x H x F_out
    # att_dst: 1 x H x F_out
    # bias: F_out
    # output: N x F_out
    # out_grad: N x F_out
    N = x.shape[0]
    F_out = weight.shape[0]
    dtype = x.dtype

    grads = {'x': None, 'lin_src.weight': None, 'bias': None,
             'att_src': None,
             'att_dst': None}

    H_prime = x @ weight.t()  # N x F_out
    alpha_src = torch.sum(H_prime * att_src[0], dim=-1)  # N x H
    alpha_dst = torch.sum(H_prime * att_dst[0], dim=-1)  # N x H

    C_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            col = A_cols[j]
            C_vals[j] = alpha_src[col] + alpha_dst[i]
    Tau_vals = torch.exp(torch.maximum(0.2 * C_vals, C_vals))
    Tau_sum = torch.zeros((N,), dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            Tau_sum[i] += Tau_vals[j]

    Phi_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            Phi_vals[j] = Tau_vals[j] / Tau_sum[i]
    # Tau_sum = torch.sum(Tau, dim=1)[:, None]  # N x 1 x H
    # Phi = Tau / Tau_sum  # N x N x H

    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            col = A_cols[j]
            assert torch.allclose(Phi_vals[j], att_weights[i, col, 0])
    # assert torch.allclose(Phi[:, :, None], att_weights)

    d_alpha = adj_matrix.t() * (out_grad @ H_prime.T)  # N x N
    d_alpha_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            col = A_cols[j]
            for k in range(F_out):
                d_alpha_vals[j] = torch.dot(out_grad[i], H_prime[col])

    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            col = A_cols[j]
            assert torch.allclose(d_alpha_vals[j], d_alpha[i, col])

    # dE = (d_alpha - dot_prods[:, None]) * att_weights[..., 0]  # N x N

    dot_prods = torch.zeros((N,), dtype=dtype)
    for i in range(N):
        dot_prods[i] = torch.dot(Phi_vals[A_rowptrs[i]:A_rowptrs[i + 1]],
                                 d_alpha_vals[
                                 A_rowptrs[i]:A_rowptrs[i + 1]])
        # for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
        #     dot_prods[i] += d_alpha_vals[j] * Phi_vals[j]

    dE_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            dE_vals[j] = (d_alpha_vals[j] - dot_prods[i]) * Phi_vals[j]

    def d_leaky_relu(x):
        return torch.where(x > 0, torch.ones_like(x),
                           torch.ones_like(x) * 0.2)

    dC_vals = dE_vals * d_leaky_relu(C_vals)

    dl = torch.zeros((N,), dtype=dtype)
    dr = torch.zeros((N,), dtype=dtype)
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            col = A_cols[j]
            dl[i] += dC_vals[j]
            dr[col] += dC_vals[j]

    dH_prime = torch.outer(dl, att_dst[0, 0]) + torch.outer(dr,
                                                            att_src[0, 0])
    # dH_prime += Phi.t @ out_grad
    for i in range(N):
        for j in range(A_rowptrs[i], A_rowptrs[i + 1]):
            for k in range(F_out):
                col = A_cols[j]
                dH_prime[col, k] += Phi_vals[j] * out_grad[i, k]

    dWeights = x.T @ dH_prime  # F_in x F_out
    d_x = dH_prime @ weight  # N x F_in

    grads['att_src'] = H_prime.T @ dr
    grads['att_dst'] = H_prime.T @ dl
    grads['lin_src.weight'] = dWeights.T
    grads['x'] = d_x
    grads['H_prime'] = dH_prime
    grads['att_weights'] = torch.tensor(
        sparse.csr_matrix((d_alpha_vals, A_cols, A_rowptrs),
                          shape=(N, N)).todense())

    grads['bias'] = torch.sum(out_grad, dim=0)
    return grads


def test_bwd_weight_compute_csc():
    N = 3
    F_in = 2
    F_out = 4
    heads = 1

    adj_matrix, layer, random_mask, x = setup_data(N, F_in, F_out, heads)
    adj_matrix_dense = adj_matrix.to_dense()
    rowptr_t, col_t, _ = adj_matrix.t().csr()

    att_weights, pred = pred_torch(adj_matrix, layer, random_mask, x)

    mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
                  att_dst=layer.att_dst)
    mygat_x = copy.deepcopy(x)
    mygat_x.grad = None
    mygat_out = mygat.forward(mygat_x, adj_matrix_dense)
    assert torch.allclose(mygat_out, pred)
    assert check_equal(mygat.att_weights.detach().numpy()[..., None],
                       att_weights.to_dense().detach().numpy())
    loss = torch.sum(mygat_out * random_mask)
    mygat.H_prime.retain_grad()
    mygat.att_weights.retain_grad()
    loss.backward()

    params = dict(layer.named_parameters())
    params['x'] = x
    params['H_prime'] = mygat.H_prime
    params['att_weights'] = mygat.att_weights

    with torch.no_grad():
        result = compute_grads_csc(x, rowptr_t, col_t, adj_matrix_dense,
                                   layer.lin_src.weight,
                                   layer.att_src, layer.att_dst,
                                   layer.bias, att_weights.to_dense(), pred,
                                   pred.grad)

    check_grads(params, result)


def compute_grads_coo(x, A_rows, A_cols, adj_matrix, weight,
                      att_src, att_dst,
                      bias,
                      att_weights,
                      output,
                      out_grad):
    # x: N x F_in
    # adj_matrix: N x N
    # weight: F_out x F_in
    # att_src: 1 x H x F_out
    # att_dst: 1 x H x F_out
    # bias: F_out
    # output: N x F_out
    # out_grad: N x F_out
    N = x.shape[0]
    nnz = A_rows.shape[0]
    F_out = weight.shape[0]
    dtype = x.dtype

    grads = {'x': None, 'lin_src.weight': None, 'bias': None,
             'att_src': None,
             'att_dst': None}

    H_prime = x @ weight.t()  # N x F_out
    alpha_src = torch.sum(H_prime * att_src[0], dim=-1)  # N x H
    alpha_dst = torch.sum(H_prime * att_dst[0], dim=-1)  # N x H

    C_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(nnz):
        C_vals[i] = alpha_src[A_rows[i]] + alpha_dst[A_cols[i]]

    Tau_vals = torch.exp(torch.maximum(0.2 * C_vals, C_vals))

    Tau_sum = torch.zeros((N,), dtype=dtype)
    for i in range(nnz):
        Tau_sum[A_cols[i]] += Tau_vals[i]

    Phi_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(nnz):
        Phi_vals[i] = Tau_vals[i] / Tau_sum[A_cols[i]]

    for i in range(nnz):
        assert torch.allclose(Phi_vals[i], att_weights[A_cols[i], A_rows[i], 0])

    d_alpha = adj_matrix.t() * (out_grad @ H_prime.T)  # N x N
    d_alpha_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        d_alpha_vals[i] = torch.dot(out_grad[col], H_prime[row])

    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        assert torch.allclose(d_alpha_vals[i], d_alpha[col, row])

    dot_prods_dense = torch.zeros((N,), dtype=dtype)
    for i in range(N):
        dot_prods_dense[i] = d_alpha[i, :] @ att_weights[i, :, 0].T  # N x N
    dE = (d_alpha - dot_prods_dense[:, None]) * att_weights[..., 0]  # N x N

    dot_prods = torch.zeros((N,), dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        dot_prods[col] += Phi_vals[i] * d_alpha_vals[i]

    for i in range(N):
        assert torch.allclose(dot_prods[i], dot_prods_dense[i])

    dE_vals = torch.zeros(A_cols.shape[0], dtype=dtype)
    for i in range(nnz):
        dE_vals[i] = (d_alpha_vals[i] - dot_prods[A_cols[i]]) * Phi_vals[i]

    for i in range(nnz):
        assert torch.allclose(dE_vals[i], dE[A_cols[i], A_rows[i]])

    def d_leaky_relu(x):
        return torch.where(x > 0, torch.ones_like(x),
                           torch.ones_like(x) * 0.2)

    dC_vals = dE_vals * d_leaky_relu(C_vals)

    dl = torch.zeros((N,), dtype=dtype)
    dr = torch.zeros((N,), dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        dl[col] += dC_vals[i]
        dr[row] += dC_vals[i]

    dH_prime = torch.outer(dl, att_dst[0, 0]) + torch.outer(dr,
                                                            att_src[0, 0])

    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        for k in range(F_out):
            dH_prime[row, k] += Phi_vals[i] * out_grad[col, k]

    dWeights = x.T @ dH_prime  # F_in x F_out
    d_x = dH_prime @ weight  # N x F_in

    grads['att_src'] = H_prime.T @ dr
    grads['att_dst'] = H_prime.T @ dl
    grads['lin_src.weight'] = dWeights.T
    grads['x'] = d_x
    grads['H_prime'] = dH_prime
    grads['att_weights'] = torch.tensor(
        sparse.coo_matrix((d_alpha_vals, (A_cols, A_rows)),
                          shape=(N, N)).todense())

    grads['bias'] = torch.sum(out_grad, dim=0)
    return grads


def test_bwd_weight_compute_coo():
    N = 3
    F_in = 2
    F_out = 4
    heads = 1

    adj_matrix, layer, random_mask, x = setup_data(N, F_in, F_out, heads)
    row, col, _ = adj_matrix.coo()
    adj_matrix_dense = adj_matrix.to_dense()

    att_weights, pred = pred_torch(adj_matrix, layer, random_mask, x)

    mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
                  att_dst=layer.att_dst, bias=layer.bias)
    mygat_x = copy.deepcopy(x)
    mygat_x.grad = None
    mygat_out = mygat.forward(mygat_x, adj_matrix_dense)
    assert torch.allclose(mygat_out, pred)
    assert check_equal(mygat.att_weights.detach().numpy()[..., None],
                       att_weights.to_dense().detach().numpy(), silent=True)
    loss = torch.sum(mygat_out * random_mask)
    mygat.H_prime.retain_grad()
    mygat.att_weights.retain_grad()
    loss.backward()

    params = dict(layer.named_parameters())
    params['x'] = x
    params['H_prime'] = mygat.H_prime
    params['att_weights'] = mygat.att_weights

    with torch.no_grad():
        result = compute_grads_coo(x, row, col, adj_matrix_dense,
                                   layer.lin_src.weight,
                                   layer.att_src, layer.att_dst,
                                   layer.bias, att_weights.to_dense(), pred,
                                   pred.grad)

    check_grads(params, result)
