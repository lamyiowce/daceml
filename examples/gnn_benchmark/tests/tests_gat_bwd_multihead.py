import copy

import pytest
import torch
from scipy import sparse

from examples.gnn_benchmark.tests.common import check_equal, check_grads, pred_torch, setup_data
from examples.gnn_benchmark.tests.tests_mygat import MyGat


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
    _, heads, F_out = att_dst.shape
    dtype = x.dtype

    grads = {'x': None, 'lin_src.weight': None, 'bias': None,
             'att_src': None,
             'att_dst': None}

    H_prime = x @ weight.t()  # N x F_out
    H_prime = torch.reshape(H_prime, (N, heads, F_out))
    alpha_src = torch.sum(H_prime * att_src, dim=-1)  # N x H
    alpha_dst = torch.sum(H_prime * att_dst, dim=-1)  # N x H

    C_vals = torch.zeros((A_cols.shape[0], heads), dtype=dtype)
    for i in range(nnz):
        C_vals[i] = alpha_src[A_rows[i]] + alpha_dst[A_cols[i]]

    Tau_vals = torch.exp(torch.maximum(0.2 * C_vals, C_vals))

    Tau_sum = torch.zeros((N, heads), dtype=dtype)
    for i in range(nnz):
        Tau_sum[A_cols[i]] += Tau_vals[i]

    Phi_vals = torch.zeros((A_cols.shape[0], heads), dtype=dtype)
    for i in range(nnz):
        Phi_vals[i] = Tau_vals[i] / Tau_sum[A_cols[i]]

    for i in range(nnz):
        assert torch.allclose(Phi_vals[i], att_weights[A_cols[i], A_rows[i]])

    out_grad = torch.reshape(out_grad, (N, heads, F_out))

    # d_alpha = adj_matrix.t() * (out_grad @ H_prime.T)  # N x N
    d_alpha = adj_matrix.t()[..., None] * torch.einsum('nhf,mhf->nmh', out_grad, H_prime)
    d_alpha_vals = torch.zeros((nnz, heads), dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        d_alpha_vals[i] = torch.einsum('hf,hf->h', out_grad[col], H_prime[row])

    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        assert torch.allclose(d_alpha_vals[i], d_alpha[col, row])

    dot_prods_dense = torch.zeros((N, heads), dtype=dtype)
    for i in range(N):
        dot_prods_dense[i] = torch.einsum('nh,nh->h', d_alpha[i], att_weights[i])  # N x N
    dE = (d_alpha - dot_prods_dense[:, None, :]) * att_weights  # N x N x H

    dot_prods = torch.zeros((N, heads), dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        dot_prods[col] += Phi_vals[i] * d_alpha_vals[i]

    for i in range(N):
        assert torch.allclose(dot_prods[i], dot_prods_dense[i])

    dE_vals = torch.zeros((nnz, heads), dtype=dtype)
    for i in range(nnz):
        dE_vals[i] = (d_alpha_vals[i] - dot_prods[A_cols[i]]) * Phi_vals[i]

    for i in range(nnz):
        assert torch.allclose(dE_vals[i], dE[A_cols[i], A_rows[i]])

    def d_leaky_relu(x):
        return torch.where(x > 0, torch.ones_like(x),
                           torch.ones_like(x) * 0.2)

    dC_vals = dE_vals * d_leaky_relu(C_vals)

    dl = torch.zeros((N, heads), dtype=dtype)
    dr = torch.zeros((N, heads), dtype=dtype)
    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        dl[col] += dC_vals[i]
        dr[row] += dC_vals[i]

    dH_prime = torch.einsum('nh,hf->nhf', dl, att_dst[0]) + torch.einsum('nh,hf->nhf', dr,
                                                                         att_src[0])

    for i in range(nnz):
        col = A_cols[i]
        row = A_rows[i]
        for k in range(F_out):
            dH_prime[row, :, k] += Phi_vals[i] * out_grad[col, :, k]

    dWeights = x.T @ torch.reshape(dH_prime, (N, heads * F_out))  # F_in x F_out
    d_x = torch.reshape(dH_prime, (N, heads * F_out)) @ weight  # N x F_in

    grads['att_src'] = torch.einsum('nhf,nh->hf', H_prime, dr)
    grads['att_dst'] = torch.einsum('nhf,nh->hf', H_prime, dl)
    grads['lin_src.weight'] = dWeights.T
    grads['x'] = d_x
    # grads['H_prime'] = torch.reshape(dH_prime, (N, heads * F_out))
    grads['H_prime'] = None
    # grads['att_weights'] = torch.tensor(
    #     sparse.coo_matrix((d_alpha_vals, (A_cols, A_rows)),
    #                       shape=(N, N)).todense())
    grads['att_weights'] = None

    grads['bias'] = torch.reshape(torch.sum(out_grad, dim=0), (heads * F_out,))
    return grads

@pytest.mark.parametrize('heads', [1, 2])
def test_bwd_weight_compute_coo(heads):
    N = 3
    F_in = 2
    F_out = 4

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
    assert check_equal(mygat.att_weights.detach().numpy(),
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
