import copy

import dace
import numpy as np
import scipy
import torch
from torch_geometric.nn import GATConv

from examples.gnn_benchmark.sparse_mm.coomm import coomm
from examples.gnn_benchmark.tests.common import check_equal, check_grads, setup_data
from examples.gnn_benchmark.tests.tests_mygat import MyGat

NEGATIVE_SLOPE = 0.2

if torch.cuda.is_available():
    import cupy as xp
    import cupyx.scipy.sparse as xps
else:
    import numpy as xp
    import scipy.sparse as xps


def test_bwd_coo_dace():
    N = 3
    F_in = 2
    F_out = 4
    heads = 2
    val_dtype = xp.float32
    negative_slope = 0.2
    torch.random.manual_seed(42)
    xp.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    _, layer, random_mask, x, graph = setup_data(N, F_in, F_out, heads)
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
        # # Transform input features.
        # Transform input features.
        features_tmp = np.einsum('ij,kj->ik', node_features,
                                 lin_srcDOTweight)
        # features: N x H x F'
        features = np.reshape(features_tmp,
                              (N, heads, num_out_features))

        # This ends up ridiculously slow because the outer loop is
        # executed on gpu and everything inside is executed
        # sequentially. The loop is moved to Sequential and the
        # inside matmul to GPU in my_auto_optimize.py.

        features_perm = np.empty((heads, N, num_out_features), dtype=dtype)
        for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
            features_perm[j, i, k] = features[i, j, k]

        alpha_src = dace.define_local((heads, N,), dtype=dtype)
        alpha_dst = dace.define_local((heads, N,), dtype=dtype)
        for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
            alpha_src[h] = features_perm[h] @ att_src[0, h]
            alpha_dst[h] = features_perm[h] @ att_dst[0, h]

        # Calculate attention weights.
        e = np.empty((heads, num_entries), dtype=dtype)
        softmax_sum = np.zeros((N, heads), dtype=dtype)

        for h, i in dace.map[0:heads, 0:num_entries]:
            row = rows[i]
            col = columns[i]
            e_tmp = alpha_src[h, row] + alpha_dst[h, col]
            # # LeakyReLU
            e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
            e_tmp = np.exp(e_tmp)
            e[h, i] = e_tmp

            # TODO: This is a workaround. With no schedule type, the results are incorrect with autoopt on CPU.
            # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
            # col = columns[i]
            softmax_sum[col, h] += e[h, i]

        # Softmax normalization.
        for h, j in dace.map[0:heads, 0:num_entries]:
            colj = columns[j]
            e[h, j] = e[h, j] / softmax_sum[colj, h]



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

    x_cpu = x.get() if device == 'cuda' else x
    rows_cpu = rows.get() if device == 'cuda' else rows
    cols_cpu = cols.get() if device == 'cuda' else cols
    with torch.no_grad():
        backward_fn.f(node_features=x_cpu, rows=rows_cpu, columns=cols_cpu,
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

    check_equal(expected_pred=np.array(mygat.H_prime.detach().cpu()), pred=H_prime_out_vanilla,
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
