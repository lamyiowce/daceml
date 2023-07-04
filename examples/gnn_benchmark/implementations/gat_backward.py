from typing import List, Optional, Tuple, Union

import dace
import numpy as np
from dace.registry import autoregister_params
from dace.sdfg import nodes as nd

from daceml.autodiff import BackwardImplementation, BackwardContext, \
    BackwardResult, utils as autodiff_utils
from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.sparse_mm.coomm import coomm


@autoregister_params(op="torch_geometric.nn.conv.gat_conv.GATConv",
                     name="coo")
class GATConvBackwardCOO(BackwardImplementation):
    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:
        output_shape = autodiff_utils.forward_out_desc_with_name(
            forward_node, context, "output").shape

        N, _ = output_shape
        node_features_desc = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features")
        F_in = node_features_desc.shape[1]

        row_shape = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "rows").shape
        num_entries = row_shape[0]

        negative_slope = forward_node.module.negative_slope
        heads = forward_node.module.heads
        F_out = forward_node.module.out_channels

        val_dtype = node_features_desc.dtype
        compute_grad_for_node_features = 'node_features' in required_gradients

        if not compute_grad_for_node_features:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad):
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

                if heads == 1:
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

                    lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', dH_prime, node_features)  # F_out x F_in
                    # node_features_grad[:] = dH_prime @ lin_srcDOTweight  # N x F_in
                    att_dst_grad[:] = features.T @ dl  # F_out
                    att_src_grad[:] = features.T @ dr  # F_out
                    bias_grad[:] = np.sum(output_grad, axis=0)
                else:
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

                    # alpha_src = dace.define_local((heads, N,), dtype=val_dtype)
                    # alpha_dst = dace.define_local((heads, N,), dtype=val_dtype)
                    alpha_src = np.zeros((heads, N,), dtype=val_dtype)
                    alpha_dst = np.zeros((heads, N,), dtype=val_dtype)
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
                    for h in range(heads):
                        coomm(A_rows=rows, A_cols=columns, A_vals=e[h], B=output_grad_perm[h],
                              C=dH_prime_perm[h],
                              beta=1.0, transA=False)

                    dH_prime = np.empty((N, heads, F_out), dtype=val_dtype)
                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] = dH_prime_perm[h, i, k]

                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] += dl[i, h] * att_dst[0, h, k]

                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] += dr[i, h] * att_src[0, h, k]

                    # dWeights = node_features.T @ np.reshape(dH_prime, (N, heads * F_out))
                    # lin_srcDOTweight_grad[:] = np.transpose(dWeights)  # head * F_out x F_in
                    reshaped_dH_prime = np.reshape(dH_prime, (N, heads * F_out))
                    lin_srcDOTweight_grad[:] = np.einsum('nm,nf->fm', node_features, reshaped_dH_prime)
                    # node_features_grad[:] = np.reshape(dH_prime,
                    #                                    (N,
                    #                                     heads * F_out)) @ lin_srcDOTweight  # N x F_in

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
        else:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad,
                            node_features_grad):
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
                output_grad: N x h * F
                """
                if heads == 1:
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

                    coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=dH_prime, beta=1.0,
                          transA=False)

                    for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dst_mult = dl[i] * att_dst[0, 0, k]
                        dH_prime[i, k] += dst_mult

                    for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        src_mult = dr[i] * att_src[0, 0, k]
                        dH_prime[i, k] += src_mult

                    lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', dH_prime, node_features)  # F_out x F_in
                    node_features_grad[:] = dH_prime @ lin_srcDOTweight  # N x F_in
                    att_dst_grad[:] = features.T @ dl  # F_out
                    att_src_grad[:] = features.T @ dr  # F_out
                    bias_grad[:] = np.sum(output_grad, axis=0)
                else:
                    ### RECOMPUTE FORWARD VALUES ###
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

                    # TODO zeros -> empty
                    d_alpha_vals = np.empty((num_entries, heads), dtype=val_dtype)
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

                    # TODO zeros -> empty
                    dE_vals = np.empty((num_entries, heads), dtype=val_dtype)
                    for i in dace.map[0:num_entries]:
                        col = columns[i]
                        dE_vals[i] = (d_alpha_vals[i] - dot_prods[col]) * e[:, i]

                    dC_vals = np.empty((num_entries, heads), dtype=val_dtype)
                    for h, i in dace.map[0:heads, 0:num_entries]:
                        dC_vals[i, h] = dE_vals[i, h] * (C_vals[i, h] > 0) + dE_vals[i, h] * (C_vals[i, h] <= 0) * negative_slope
                    # dC_vals = dE_vals * (C_vals > 0) + dE_vals * (C_vals <= 0) * negative_slope

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

                    output_grad_perm = np.empty((heads, N, F_out), dtype=val_dtype)
                    for i, h, k in dace.map[0:num_entries, 0:heads, 0:F_out]:
                        output_grad_perm[h, i, k] = output_grad_heads[i, h, k]
                    # np.transpose(output_grad_heads, (1, 0, 2))
                    for h in range(heads):
                        coomm(A_rows=rows, A_cols=columns, A_vals=e[h], B=output_grad_perm[h],
                              C=dH_prime_perm[h],
                              beta=1.0, transA=False)

                    dH_prime = np.empty((N, heads, F_out), dtype=val_dtype)
                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] = dH_prime_perm[h, i, k]

                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] += dl[i, h] * att_dst[0, h, k]

                    for h, i, k in dace.map[0:heads, 0:N,
                                   0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                        dH_prime[i, h, k] += dr[i, h] * att_src[0, h, k]

                    reshaped_dH_prime = np.reshape(dH_prime, (N, heads * F_out))
                    lin_srcDOTweight_grad[:] = np.einsum('nm,nf->fm', node_features, reshaped_dH_prime)
                    node_features_grad[:] = reshaped_dH_prime @ lin_srcDOTweight  # N x F_in

                    for h, k in dace.map[0:heads, 0:F_out]:
                        att_dst_grad[0, h, k] = 0
                        for n in dace.map[0:N]:
                            att_dst_grad[0, h, k] += dl[n, h] * features[n, h, k]

                    for h, k in dace.map[0:heads, 0:F_out]:
                        att_src_grad[0, h, k] = 0
                        for n in dace.map[0:N]:
                            att_src_grad[0, h, k] += dr[n, h] * features[n, h, k]
                    bias_grad[:] = np.reshape(np.sum(output_grad, axis=0), (heads * F_out,))

        result_node, result = autodiff_utils.backward_program_for_node(
            backward_fn, context, forward_node)

        return result_node, result
