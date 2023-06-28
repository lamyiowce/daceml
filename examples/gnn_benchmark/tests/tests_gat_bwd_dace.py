import copy

import dace
import numpy as np
import scipy
import torch
from torch_geometric.nn import GATConv

from examples.gnn_benchmark.sparse_mm.coomm import coomm
from examples.gnn_benchmark.tests.common import check_equal
from examples.gnn_benchmark.tests.tests_gat_bwd import check_grads

NEGATIVE_SLOPE = 0.2

if torch.cuda.is_available():
    import cupy as xp
    import cupyx.scipy.sparse as xps
else:
    import numpy as xp
    import scipy.sparse as xps


def setup_data(N, F_in, F_out, heads=1, seed=42):
    dtype = xp.float32
    torch.random.manual_seed(seed)
    xp.random.seed(seed)
    layer = GATConv(F_in, F_out, heads=heads, add_self_loops=False,
                    negative_slope=NEGATIVE_SLOPE, bias=True)
    x = xp.random.randn(N, F_in).astype(dtype)
    graph = xps.random(N, N, density=0.5, format='csr', dtype=dtype)
    graph.data = xp.ones_like(graph.data)
    random_mask = xp.reshape(xp.arange(N * F_out), (N, F_out)).astype(dtype) / 10

    return x, graph, random_mask, layer


class MyGat(torch.nn.Module):
    def __init__(self, weight, att_src, att_dst, bias=None):
        super().__init__()
        self.weight = weight.detach().clone()
        print(weight.device)
        print(self.weight.device)
        self.att_src = att_src.detach().clone()
        self.att_dst = att_dst.detach().clone()
        if bias is not None:
            self.bias = bias.detach().clone()

    def forward(self, x, adj_matrix):
        self.H_prime = x @ self.weight.t()  # N x F_out
        self.H_prime.retain_grad()
        alpha_src = torch.sum(self.H_prime * self.att_src[0], dim=-1)  # N x H
        alpha_dst = torch.sum(self.H_prime * self.att_dst[0], dim=-1)  # N x H
        C = (alpha_src[None, :] + alpha_dst[:, None])  # N x N x H
        Tau = adj_matrix.t() * torch.exp(torch.maximum(0.2 * C, C))
        Tau_sum = torch.sum(Tau, dim=1)[:, None]  # N x 1 x H
        self.att_weights = Tau / Tau_sum  # N x N x H
        self.att_weights.retain_grad()
        Z = (adj_matrix.t() * self.att_weights) @ self.H_prime  # N x H
        return Z + self.bias


def test_bwd_coo_dace():
    N = 3
    F_in = 2
    F_out = 4
    heads = 1
    val_dtype = xp.float32
    negative_slope = 0.2
    torch.random.manual_seed(42)
    xp.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    x, graph, random_mask, layer = setup_data(N, F_in, F_out, heads)
    layer = layer.to(device)
    num_entries = graph.nnz
    graph.data = xp.ones_like(graph.data)
    rows, cols = graph.tocoo().row, graph.tocoo().col

    @dace.program
    def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                    att_src, att_dst, att_src_grad, att_dst_grad,
                    lin_srcDOTweight_grad, bias_grad, output_grad,
                    node_features_grad,
                    H_prime_grad, att_weights_grad,
                    att_weights_out, H_prime_out):
        """
        node_features: input features, N x M
        rows: rows, K
        columns: col, K
        edge_vals: values, K
        linDOTweight: F x M
        bias: F
        output: N x F

        node_features_grad: N x M
        linDOTweight_grad: F x M
        output_grad: N x F
        """

        ### RECOMPUTE FORWARD VALUES ###
        # Transform input features.
        features = np.einsum('ij,kj->ik', node_features,
                             lin_srcDOTweight)
        alpha_src = features @ att_src[0, 0]
        alpha_dst = features @ att_dst[0, 0]

        # Calculate attention weights.
        C_vals = np.empty((num_entries,), dtype=val_dtype)
        e = np.empty((num_entries,), dtype=val_dtype)
        softmax_sum = np.zeros((N,), dtype=val_dtype)

        for i in dace.map[0:num_entries]:
            row = rows[i]
            col = columns[i]
            e_tmp = alpha_src[row] + alpha_dst[col]
            C_vals[i] = e_tmp
            # # LeakyReLU
            e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
            e_tmp = np.exp(e_tmp)
            e[i] = e_tmp

            # # TODO: This is a workaround. With no schedule type, the results are incorrect
            #  on CPU with autoopt.
            # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
            # col = columns[i]
            softmax_sum[col] += e[i]

        # Softmax normalization.
        for j in dace.map[0:num_entries]:
            colj = columns[j]
            e[j] = e[j] / softmax_sum[colj]

        ### COMPUTE THE GRADIENTS ###
        d_alpha_vals = np.zeros((num_entries,), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            row = rows[i]
            d_alpha_vals[i] = np.dot(output_grad[col], features[row])

        dot_prods = np.zeros((N,), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            dot_prods[col] += e[i] * d_alpha_vals[i]

        dE_vals = np.zeros((num_entries,), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            dE_vals[i] = (d_alpha_vals[i] - dot_prods[col]) * e[i]

        dC_vals = dE_vals * (C_vals > 0) + dE_vals * (C_vals <= 0) * negative_slope

        dl = np.zeros((N,), dtype=val_dtype)
        dr = np.zeros((N,), dtype=val_dtype)

        # Generates an incorrect SDFG!!!!
        # for i in dace.map[0:num_entries]:
        #     col = columns[i]
        #     row = rows[i]
        #     dl[col] += dC_vals[i]
        #     dr[row] += dC_vals[i]

        for i in dace.map[0:num_entries]:
            col = columns[i]
            dl[col] += dC_vals[i]

        for i in dace.map[0:num_entries]:
            row = rows[i]
            dr[row] += dC_vals[i]

        dH_prime = np.zeros((N, F_out), dtype=val_dtype)

        # for i, k in dace.map[0:num_entries, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
        #     col = columns[i]
        #     mult = e[i] * output_grad[col, k]
        #     row = rows[i]
        #     dH_prime[row, k] += mult
        coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=dH_prime, beta=1.0,
              transA=False)

        for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, k] += dl[i] * att_dst[0, 0, k]

        for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, k] += dr[i] * att_src[0, 0, k]

        att_weights_grad[:] = d_alpha_vals  # K
        H_prime_grad[:] = dH_prime  # N x F_out
        lin_srcDOTweight_grad[:] = dH_prime.T @ node_features  # F_out x F_in
        node_features_grad[:] = dH_prime @ lin_srcDOTweight  # N x F_in
        att_dst_grad[:] = features.T @ dl  # F_out
        att_src_grad[:] = features.T @ dr  # F_out
        bias_grad[:] = np.sum(output_grad, axis=0)
        att_weights_out[:] = e  # K
        H_prime_out[:] = features  # N x F_out

    mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
                  att_dst=layer.att_dst, bias=layer.bias)
    mygat_x = torch.tensor(x, requires_grad=True, device=device)
    mygat_out = mygat.forward(mygat_x, torch.tensor(graph.todense(), device=device))

    loss = torch.sum(mygat_out * torch.tensor(random_mask, device=device))
    mygat_out.retain_grad()
    mygat.H_prime.retain_grad()
    mygat.att_weights.retain_grad()
    loss.backward()

    params = dict(mygat.named_parameters())
    params['x'] = mygat_x
    params['H_prime'] = mygat.H_prime
    params['att_weights'] = mygat.att_weights

    weight_grad = np.zeros((F_out, F_in), dtype=val_dtype)
    bias_grad = np.zeros((F_out,), dtype=val_dtype)
    att_src_grad = np.zeros((1, 1, F_out), dtype=val_dtype)
    att_dst_grad = np.zeros((1, 1, F_out), dtype=val_dtype)
    x_grad = np.zeros((N, F_in), dtype=val_dtype)
    H_prime_grad = np.zeros((N, F_out), dtype=val_dtype)
    att_weights_grad = np.zeros((num_entries,), dtype=val_dtype)

    att_weights_out_vanilla = np.zeros((num_entries,), dtype=val_dtype)
    H_prime_out_vanilla = np.zeros((N, F_out), dtype=val_dtype)

    with torch.no_grad():
        backward_fn.f(node_features=x.get(), rows=rows.get(), columns=cols.get(),
                      lin_srcDOTweight=layer.lin_src.weight.detach().cpu().numpy(),
                      att_src=layer.att_src.detach().cpu().numpy(),
                      att_dst=layer.att_dst.detach().cpu().numpy(),
                      output_grad=mygat_out.grad.detach().cpu().numpy(),
                      att_weights_out=att_weights_out_vanilla,
                      H_prime_out=H_prime_out_vanilla,
                      node_features_grad=x_grad,
                      lin_srcDOTweight_grad=weight_grad,
                      bias_grad=bias_grad,
                      att_src_grad=att_src_grad,
                      att_dst_grad=att_dst_grad,
                      att_weights_grad=att_weights_grad,
                      H_prime_grad=H_prime_grad)

    vanilla_result = {}
    vanilla_result['x'] = xp.copy(x_grad)
    vanilla_result['H_prime'] = xp.copy(H_prime_grad)
    print(att_weights_grad.shape, rows.shape, cols.shape)
    vanilla_result['att_weights'] = xp.copy(
        xps.coo_matrix((xp.array(att_weights_grad), (cols, rows)), shape=(N, N)).todense())
    vanilla_result['weight'] = xp.copy(weight_grad)
    vanilla_result['bias'] = xp.copy(bias_grad)
    vanilla_result['att_src'] = xp.copy(att_src_grad)
    vanilla_result['att_dst'] = xp.copy(att_dst_grad)

    check_equal(expected_pred=xp.array(mygat.H_prime.detach()), pred=H_prime_out_vanilla,
                name='H_prime')
    check_equal(expected_pred=xp.array(mygat.att_weights.detach()),
                pred=xps.coo_matrix((xp.array(att_weights_out_vanilla), (cols, rows)),
                                    shape=(N, N)).todense(),
                name='att_weights')

    check_grads(expected_params=params, result=vanilla_result)
    print("VANILLA FN OK!")

    weight_grad = xp.zeros((F_out, F_in), dtype=val_dtype)
    bias_grad = xp.zeros((F_out,), dtype=val_dtype)
    att_src_grad = xp.zeros((1, 1, F_out), dtype=val_dtype)
    att_dst_grad = xp.zeros((1, 1, F_out), dtype=val_dtype)
    x_grad = xp.zeros_like(x, dtype=val_dtype)
    H_prime_grad = xp.zeros((N, F_out), dtype=val_dtype)
    att_weights_grad = xp.zeros((num_entries,), dtype=val_dtype)

    att_weights_out = xp.zeros((num_entries,), dtype=val_dtype)
    H_prime_out = xp.zeros((N, F_out), dtype=val_dtype)
    with torch.no_grad():
        sdfg = backward_fn.to_sdfg(node_features=x, rows=rows, columns=cols,
                                   lin_srcDOTweight=xp.array(layer.lin_src.weight.cpu()),
                                   att_src=xp.array(layer.att_src.cpu()),
                                   att_dst=xp.array(layer.att_dst.cpu()),
                                   output_grad=xp.array(mygat_out.grad.cpu()),
                                   att_weights_out=att_weights_out,
                                   H_prime_out=H_prime_out,
                                   node_features_grad=x_grad,
                                   lin_srcDOTweight_grad=weight_grad,
                                   bias_grad=bias_grad,
                                   att_src_grad=att_src_grad,
                                   att_dst_grad=att_dst_grad,
                                   att_weights_grad=att_weights_grad,
                                   H_prime_grad=H_prime_grad)
        if torch.cuda.is_available():
            sdfg.apply_gpu_transformations()

        sdfg.openmp_sections = False
        sdfg(node_features=x, rows=rows, columns=cols,
             lin_srcDOTweight=xp.array(layer.lin_src.weight.cpu()),
             att_src=xp.array(layer.att_src.cpu()),
             att_dst=xp.array(layer.att_dst.cpu()),
             output_grad=xp.array(mygat_out.grad.cpu()),
             att_weights_out=att_weights_out,
             H_prime_out=H_prime_out,
             node_features_grad=x_grad,
             lin_srcDOTweight_grad=weight_grad,
             bias_grad=bias_grad,
             att_src_grad=att_src_grad,
             att_dst_grad=att_dst_grad,
             att_weights_grad=att_weights_grad,
             H_prime_grad=H_prime_grad)

    result = {}
    result['x'] = x_grad
    result['H_prime'] = H_prime_grad
    result['att_weights'] = xps.coo_matrix((att_weights_grad, (cols, rows)), shape=(N, N)).todense()
    result['weight'] = weight_grad
    result['bias'] = bias_grad
    result['att_src'] = att_src_grad
    result['att_dst'] = att_dst_grad

    check_equal(expected_pred=H_prime_out_vanilla, pred=H_prime_out, name='H_prime out')
    check_equal(expected_pred=att_weights_out_vanilla,
                pred=att_weights_out,
                name='att_weights oput')

    check_grads(expected_params=vanilla_result, result=result)


if __name__ == '__main__':
    test_bwd_coo_dace()
    print('OK!')
