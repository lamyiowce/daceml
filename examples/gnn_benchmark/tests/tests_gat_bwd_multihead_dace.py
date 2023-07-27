import copy

import dace
import numpy as np
import scipy
import torch
from torch_geometric.nn import GATConv

from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.sparse_mm.coomm import coomm
from examples.gnn_benchmark.tests.common import check_equal, check_grads, setup_data
from examples.gnn_benchmark.tests.tests_mygat import MyGat

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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
    torch.random.manual_seed(43)
    xp.random.seed(43)
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
        linDOTweight: H * F x M
        bias: H * F
        output: N x H * F

        node_features_grad: N x M
        linDOTweight_grad: H * F x M
        output_grad: N x H * F
        """

        ### RECOMPUTE FORWARD VALUES ###
        # # Transform input features.
        # Transform input features.
        features_tmp = np.einsum('ij,kj->ik', node_features,
                                 lin_srcDOTweight)
        # features: N x H x F'
        features = np.reshape(features_tmp,
                              (N, heads, F_out))

        # This ends up ridiculously slow because the outer loop is
        # executed on gpu and everything inside is executed
        # sequentially. The loop is moved to Sequential and the
        # inside matmul to GPU in my_auto_optimize.py.

        features_perm = np.empty((heads, N, F_out), dtype=val_dtype)
        for j, i, k in dace.map[0:heads, 0:N, 0:F_out]:
            features_perm[j, i, k] = features[i, j, k]

        alpha_src = dace.define_local((heads, N,), dtype=val_dtype)
        alpha_dst = dace.define_local((heads, N,), dtype=val_dtype)
        for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
            alpha_src[h] = features_perm[h] @ att_src[0, h]
            alpha_dst[h] = features_perm[h] @ att_dst[0, h]

        # Calculate attention weights.
        e = np.empty((heads, num_entries), dtype=val_dtype)
        softmax_sum = np.zeros((N, heads), dtype=val_dtype)
        C_vals = np.empty((num_entries, heads), dtype=val_dtype)

        for h, i in dace.map[0:heads, 0:num_entries]:
            row = rows[i]
            col = columns[i]
            e_tmp = alpha_src[h, row] + alpha_dst[h, col]
            C_vals[i, h] = e_tmp
            # # LeakyReLU
            e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
            e_tmp = np.exp(e_tmp)
            e[h, i] = e_tmp
            softmax_sum[col, h] += e[h, i]

        # Softmax normalization.
        for h, j in dace.map[0:heads, 0:num_entries]:
            colj = columns[j]
            e[h, j] = e[h, j] / softmax_sum[colj, h]

        ### COMPUTE THE GRADIENTS ###
        output_grad_heads = np.reshape(output_grad, (N, heads, F_out))

        d_alpha_vals = np.zeros((num_entries, heads), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            row = rows[i]
            # d_alpha_vals[i] = np.einsum('hf,hf->h', output_grad_heads[col], features[row])
            for h in dace.map[0:heads]:
                d_alpha_vals[i, h] = np.dot(output_grad_heads[col, h],
                                            features[row, h])

        dot_prods = np.zeros((N, heads), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            dot_prods[col] += e[:, i] * d_alpha_vals[i]

        dE_vals = np.zeros((num_entries, heads), dtype=val_dtype)
        for i in dace.map[0:num_entries]:
            col = columns[i]
            dE_vals[i] = (d_alpha_vals[i] - dot_prods[col]) * e[:, i]

        dC_vals = dE_vals * (C_vals > 0) + dE_vals * (C_vals <= 0) * negative_slope

        dl = np.zeros((N, heads), dtype=val_dtype)
        dr = np.zeros((N, heads), dtype=val_dtype)

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

        dH_prime_perm = np.zeros((heads, N, F_out), dtype=val_dtype)

        output_grad_perm = np.transpose(output_grad_heads, (1, 0, 2))
        for i in range(heads):
            coomm(A_rows=rows, A_cols=columns, A_vals=e[i], B=output_grad_perm[i],
                  C=dH_prime_perm[i],
                  beta=1.0, transA=False)

        dH_prime = np.zeros((N, heads, F_out), dtype=val_dtype)
        for h, i, k in dace.map[0:heads, 0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] = dH_prime_perm[h, i, k]

        for h, i, k in dace.map[0:heads, 0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] += dl[i, h] * att_dst[0, h, k]

        for h, i, k in dace.map[0:heads, 0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] += dr[i, h] * att_src[0, h, k]

        att_weights_grad[:] = d_alpha_vals  # K
        H_prime_grad[:] = dH_prime  # N x F_out

        lin_srcDOTweight_grad[:] = np.transpose(
            node_features.T @ np.reshape(dH_prime, (N, heads * F_out)))  # head * F_out x F_in
        node_features_grad[:] = np.reshape(dH_prime,
                                           (N, heads * F_out)) @ lin_srcDOTweight  # N x F_in

        for h, k in dace.map[0:heads, 0:F_out]:
            att_dst_grad[0, h, k] = 0
            for n in dace.map[0:N]:
                att_dst_grad[0, h, k] += dl[n, h] * features[n, h, k]

        for h, k in dace.map[0:heads, 0:F_out]:
            att_src_grad[0, h, k] = 0
            for n in dace.map[0:N]:
                att_src_grad[0, h, k] += dr[n, h] * features[n, h, k]
        # att_dst_grad[:] = np.einsum('nhf,nh->hf', features, dl)  # F_out
        # att_src_grad[:] = np.einsum('nhf,nh->hf', features, dr)  # F_out
        bias_grad[:] = np.reshape(np.sum(output_grad, axis=0), (heads * F_out,))
        att_weights_out[:] = e.T  # K
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

    weight_grad = np.zeros((F_out * heads, F_in), dtype=val_dtype)
    bias_grad = np.zeros((F_out * heads,), dtype=val_dtype)
    att_src_grad = np.zeros((1, heads, F_out), dtype=val_dtype)
    att_dst_grad = np.zeros((1, heads, F_out), dtype=val_dtype)
    x_grad = np.zeros((N, F_in), dtype=val_dtype)
    H_prime_grad = np.zeros((N, heads, F_out), dtype=val_dtype)
    att_weights_grad = np.zeros((num_entries, heads), dtype=val_dtype)

    att_weights_out_vanilla = np.zeros((num_entries, heads), dtype=val_dtype)
    H_prime_out_vanilla = np.zeros((N, heads, F_out), dtype=val_dtype)

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
    vanilla_result['att_weights'] = None
    for h in range(heads):
        vanilla_result[f'att_weights_{h}'] = xp.copy(
            xps.coo_matrix((xp.array(att_weights_grad[:, h]), (cols, rows)),
                           shape=(N, N)).todense())
    vanilla_result['weight'] = xp.copy(weight_grad)
    vanilla_result['bias'] = xp.copy(bias_grad)
    vanilla_result['att_src'] = xp.copy(att_src_grad)
    vanilla_result['att_dst'] = xp.copy(att_dst_grad)

    check_equal(expected_pred=np.array(mygat.H_prime.detach().cpu()), pred=H_prime_out_vanilla,
                name='H_prime')
    for h in range(heads):
        check_equal(expected_pred=xp.array(mygat.att_weights.detach()[..., h]),
                    pred=xps.coo_matrix((xp.array(att_weights_out_vanilla[:, h]), (cols, rows)),
                                        shape=(N, N)).todense(),
                    name=f'att_weights_{h}')

    check_grads(expected_params=params, result=vanilla_result)
    print("VANILLA FN OK!")

    weight_grad = xp.zeros((F_out * heads, F_in), dtype=val_dtype)
    bias_grad = xp.zeros((F_out * heads,), dtype=val_dtype)
    att_src_grad = xp.zeros((1, heads, F_out), dtype=val_dtype)
    att_dst_grad = xp.zeros((1, heads, F_out), dtype=val_dtype)
    x_grad = xp.zeros((N, F_in), dtype=val_dtype)
    H_prime_grad = xp.zeros((N, heads, F_out), dtype=val_dtype)
    att_weights_grad = xp.zeros((num_entries, heads), dtype=val_dtype)

    att_weights_out = xp.zeros((num_entries, heads), dtype=val_dtype)
    H_prime_out = xp.zeros((N, heads, F_out), dtype=val_dtype)

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
    for h in range(heads):
        result[f'att_weights_{h}'] = xp.copy(
            xps.coo_matrix((xp.array(att_weights_grad[:, h]), (cols, rows)),
                           shape=(N, N)).todense())
    result['att_weights'] = None
    result['weight'] = weight_grad
    result['bias'] = bias_grad
    result['att_src'] = att_src_grad
    result['att_dst'] = att_dst_grad

    check_equal(expected_pred=H_prime_out_vanilla, pred=H_prime_out, name='H_prime out')
    check_equal(expected_pred=att_weights_out_vanilla,
                pred=att_weights_out,
                name='att_weights oput')

    check_grads(expected_params=vanilla_result, result=result)


def test_bwd_csr_dace():
    N = 3
    F_in = 2
    F_out = 4
    heads = 2
    val_dtype = xp.float32
    negative_slope = 0.2
    one_min_neg_slope = 1 - negative_slope
    torch.random.manual_seed(43)
    xp.random.seed(43)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, layer, random_mask, x, graph = setup_data(N, F_in, F_out, heads)
    layer = layer.to(device)
    num_entries = graph.nnz
    graph.data = xp.ones_like(graph.data)
    rowptrs, cols = xp.copy(graph.tocsr().indptr), xp.copy(graph.tocsr().indices)

    @dace.program
    def backward_fn(node_features, rowptrs, columns, lin_srcDOTweight,
                    att_src, att_dst, att_src_grad, att_dst_grad,
                    lin_srcDOTweight_grad, bias_grad, output_grad,
                    node_features_grad,
                    H_prime_grad, att_weights_grad,
                    features_saved, e, is_pos_C_vals):
        """
        node_features: input features, N x M
        rows: rows, K
        columns: col, K
        linDOTweight: H * F x M
        bias: H * F
        output: N x H * F

        node_features_grad: N x M
        linDOTweight_grad: H * F x M
        output_grad: N x H * F
        """
        ### COMPUTE THE GRADIENTS ###
        output_grad_heads = np.reshape(output_grad, (N, heads, F_out))
        # bias_grad[:] = np.sum(output_grad, axis=0)

        # SDDMM
        d_alpha_vals = np.empty((heads, num_entries), dtype=val_dtype)

        for h, i in dace.map[0:heads, 0:num_entries]:
            d_alpha_vals[h, i] = 0

        for row in dace.map[0:N] @ dace.dtypes.ScheduleType.Sequential:
            for inner_i in dace.map[
                           rowptrs[row]:rowptrs[row + 1]] @ dace.dtypes.ScheduleType.Sequential:
                for h, k in dace.map[0:heads, 0:F_out]:
                    col = columns[inner_i]
                    d_alpha_vals[h, inner_i] += output_grad_heads[col, h, k] * features_saved[
                        h, row, k]

        dot_prods = np.zeros((N, heads), dtype=val_dtype)
        for h, i in dace.map[0:heads, 0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
            col = columns[i]
            dot_prods[col, h] += e[h, i] * d_alpha_vals[h, i]

        dl = np.zeros((heads, N), dtype=val_dtype)
        dr = np.zeros((heads, N), dtype=val_dtype)
        neg_slope = dace.define_local_scalar(val_dtype)
        neg_slope[:] = negative_slope
        for row in dace.map[0:N] @ dace.dtypes.ScheduleType.Sequential:
            for i in dace.map[rowptrs[row]:rowptrs[row + 1]] @ dace.dtypes.ScheduleType.Sequential:
                for h in dace.map[0:heads]:
                    col = columns[i]
                    dE_val = dace.define_local_scalar(val_dtype)
                    dE_val[:] = (d_alpha_vals[h, i] - dot_prods[col, h]) * e[h, i]
                    dC_val = dace.define_local_scalar(val_dtype)
                    dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * is_pos_C_vals[h, i])
                    dr[h, row] += dC_val
                    dl[h, col] += dC_val

        dH_prime_perm = np.zeros((heads, N, F_out), dtype=val_dtype)

        output_grad_perm = np.transpose(output_grad_heads, (1, 0, 2))
        for h in range(heads):
            csrmm(rowptrs, columns, e[h], output_grad_perm[h], dH_prime_perm[h], beta=1.0,
                  transA=False)

        dH_prime = np.zeros((N, heads, F_out), dtype=val_dtype)
        for h, i, k in dace.map[0:heads, 0:N,
                       0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] = dH_prime_perm[h, i, k]

        for h, i, k in dace.map[0:heads, 0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] += dl[h, i] * att_dst[0, h, k]

        for h, i, k in dace.map[0:heads, 0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
            dH_prime[i, h, k] += dr[h, i] * att_src[0, h, k]

        out_reshaped_dH_prime = np.reshape(dH_prime, (N, heads * F_out))
        # This has to be np.einsum('nf,nm->fm', out_reshaped_dH_prime, node_features),
        # not np.einsum('nm,nf->fm', node_features, out_reshaped_dH_prime), because
        # the latter is bugged for f = m.
        lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', out_reshaped_dH_prime,
                                             node_features)

        for h in dace.map[0:heads]:
            att_src_grad[0, h, :] = dr[h] @ features_saved[h]  # N times N x F_out = F_out
            att_dst_grad[0, h, :] = dl[h] @ features_saved[h]

        H_prime_grad[:] = dH_prime
        att_weights_grad[:] = d_alpha_vals

    # mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
    #               att_dst=layer.att_dst, bias=layer.bias)
    # mygat_x = torch.tensor(x, requires_grad=True, device=device)
    # mygat_out = mygat.forward(mygat_x, torch.tensor(graph.todense(), device=device))
    #
    # loss = torch.sum(mygat_out * torch.tensor(random_mask, device=device))
    # mygat_out.retain_grad()
    # mygat.H_prime.retain_grad()
    # mygat.att_weights.retain_grad()
    # loss.backward()

    # params = dict(mygat.named_parameters())
    # params['x'] = mygat_x
    # params['H_prime'] = mygat.H_prime
    # params['att_weights'] = mygat.att_weights

    # random grads:
    mygat_out_grad = np.random.rand(N, F_out * heads)
    e = np.random.rand(heads, num_entries)
    is_pos_C_vals = np.random.rand(heads, num_entries).astype(np.bool)
    features_saved = np.random.rand(heads, N, F_out)


    weight_grad = np.zeros((F_out * heads, F_in), dtype=val_dtype)
    bias_grad = np.zeros((F_out * heads,), dtype=val_dtype)
    att_src_grad = np.zeros((1, heads, F_out), dtype=val_dtype)
    att_dst_grad = np.zeros((1, heads, F_out), dtype=val_dtype)
    x_grad = np.zeros((N, F_in), dtype=val_dtype)
    H_prime_grad = np.zeros((N, heads, F_out), dtype=val_dtype)
    att_weights_grad = np.zeros((heads, num_entries), dtype=val_dtype)

    att_weights_out_vanilla = np.zeros((num_entries, heads), dtype=val_dtype)
    H_prime_out_vanilla = np.zeros((N, heads, F_out), dtype=val_dtype)

    x_cpu = x.get() if device == 'cuda' else x
    rows_cpu = rowptrs.get() if device == 'cuda' else rowptrs
    cols_cpu = cols.get() if device == 'cuda' else cols
    with torch.no_grad():
        backward_fn.f(node_features=x_cpu, rowptrs=rows_cpu, columns=cols_cpu,
                      lin_srcDOTweight=layer.lin_src.weight.detach().cpu().numpy(),
                      att_src=layer.att_src.detach().cpu().numpy(),
                      att_dst=layer.att_dst.detach().cpu().numpy(),

                      output_grad=mygat_out_grad,
                      e=e,
                      is_pos_C_vals=is_pos_C_vals,
                      features_saved=features_saved,

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
    print(att_weights_grad.shape, rowptrs.shape, cols.shape)
    for h in range(heads):
        vanilla_result[f'att_weights_{h}'] = xp.copy(
            xps.csr_matrix((xp.array(att_weights_grad[h, :]), cols, rowptrs),
                           shape=(N, N)).todense())
    vanilla_result['weight'] = xp.copy(weight_grad)
    vanilla_result['bias'] = xp.copy(bias_grad)
    vanilla_result['att_src'] = xp.copy(att_src_grad)
    vanilla_result['att_dst'] = xp.copy(att_dst_grad)

    # check_equal(expected_pred=np.array(mygat.H_prime.detach().cpu()), pred=H_prime_out_vanilla,
    #             name='H_prime')
    # for h in range(heads):
    #     check_equal(expected_pred=xp.array(mygat.att_weights.detach()[..., h]),
    #                 pred=xps.coo_matrix((xp.array(att_weights_out_vanilla[:, h]), (cols, rows)),
    #                                     shape=(N, N)).todense(),
    #                 name=f'att_weights_{h}')
    #
    # check_grads(expected_params=params, result=vanilla_result)
    # print("VANILLA FN OK!")

    weight_grad = xp.zeros((F_out * heads, F_in), dtype=val_dtype)
    bias_grad = xp.zeros((F_out * heads,), dtype=val_dtype)
    att_src_grad = xp.zeros((1, heads, F_out), dtype=val_dtype)
    att_dst_grad = xp.zeros((1, heads, F_out), dtype=val_dtype)
    x_grad = xp.zeros((N, F_in), dtype=val_dtype)
    H_prime_grad = xp.zeros((N, heads, F_out), dtype=val_dtype)
    att_weights_grad = xp.zeros((heads, num_entries), dtype=val_dtype)

    att_weights_out = xp.zeros((num_entries, heads), dtype=val_dtype)
    H_prime_out = xp.zeros((N, heads, F_out), dtype=val_dtype)

    with torch.no_grad():
        sdfg = backward_fn.to_sdfg(node_features=x, rowptrs=rowptrs, columns=cols,
                                   lin_srcDOTweight=xp.array(layer.lin_src.weight.cpu()),
                                   att_src=xp.array(layer.att_src.cpu()),
                                   att_dst=xp.array(layer.att_dst.cpu()),

                                   # output_grad=xp.array(mygat_out.grad.cpu()),
                                   output_grad=mygat_out_grad,
                                   e=e,
                                   is_pos_C_vals=is_pos_C_vals,
                                   features_saved=features_saved,

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
        sdfg(node_features=x, rowptrs=rowptrs, columns=cols,
             lin_srcDOTweight=xp.array(layer.lin_src.weight.cpu()),
             att_src=xp.array(layer.att_src.cpu()),
             att_dst=xp.array(layer.att_dst.cpu()),
             # output_grad=xp.array(mygat_out.grad.cpu()),

             output_grad=mygat_out_grad,
             e=e,
             is_pos_C_vals=is_pos_C_vals,
             features_saved=features_saved,

             # att_weights_out=att_weights_out,
             # H_prime_out=H_prime_out,
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
    for h in range(heads):
        result[f'att_weights_{h}'] = xp.copy(
            xps.csr_matrix((xp.array(att_weights_grad[h, :]), cols, rowptrs),
                           shape=(N, N)).todense())
    result['att_weights'] = None
    result['weight'] = weight_grad
    result['bias'] = bias_grad
    result['att_src'] = att_src_grad
    result['att_dst'] = att_dst_grad

    # check_equal(expected_pred=H_prime_out_vanilla, pred=H_prime_out, name='H_prime out')
    # check_equal(expected_pred=att_weights_out_vanilla,
    #             pred=att_weights_out,
    #             name='att_weights oput')

    check_grads(expected_params=vanilla_result, result=result)


if __name__ == '__main__':
    test_bwd_csr_dace()
    print('OK!')
