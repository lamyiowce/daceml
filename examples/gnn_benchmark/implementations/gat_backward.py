from typing import List, Optional, Tuple, Union

import dace
import numpy as np
from dace.registry import autoregister_params
from dace.sdfg import nodes as nd
from daceml.autodiff.utils import connect_output_from_forward

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
        one_min_neg_slope = 1 - negative_slope
        heads = forward_node.module.heads
        F_out = forward_node.module.out_channels

        val_dtype = node_features_desc.dtype
        compute_grad_for_node_features = 'node_features' in required_gradients

        # if not compute_grad_for_node_features:
        @dace.program
        def basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                               att_src, att_dst, att_src_grad, att_dst_grad,
                               lin_srcDOTweight_grad, bias_grad, output_grad,
                               out_reshaped_dH_prime):
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
                C_vals = np.empty((num_entries,), dtype=dace.bool)
                e = np.empty((num_entries,), dtype=val_dtype)
                softmax_sum = np.zeros((N,), dtype=val_dtype)

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    C_vals[i] = e_tmp > 0
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

                dl = np.zeros((N,), dtype=val_dtype)
                dr = np.zeros((N,), dtype=val_dtype)
                neg_slope = dace.define_local_scalar(val_dtype)
                neg_slope[:] = negative_slope
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    row = rows[i]
                    dE_val = dace.define_local_scalar(val_dtype)
                    dE_val[:] = (d_alpha_vals[i] - dot_prods[col]) * e[i]

                    dC_val = dace.define_local_scalar(val_dtype)
                    # dC_val[:] = dE_val * (C_vals[i] > 0) + dE_val * (
                    #         C_vals[i] <= 0) * neg_slope
                    dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * C_vals[i])
                    dr[row] += dC_val
                    dl[col] += dC_val

                out_reshaped_dH_prime[:] = np.zeros((N, F_out), dtype=val_dtype)

                coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=out_reshaped_dH_prime,
                      beta=1.0,
                      transA=False)

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    out_reshaped_dH_prime[i, k] += dl[i] * att_dst[0, 0, k]

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    out_reshaped_dH_prime[i, k] += dr[i] * att_src[0, 0, k]

                lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', out_reshaped_dH_prime,
                                                     node_features)  # F_out x F_in
                att_dst_grad[:] = dl @ features  # F_out
                att_src_grad[:] = dr @ features  # F_out
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

                features_perm = np.transpose(features, (1, 0, 2))

                alpha_src = dace.define_local((heads, N,), dtype=val_dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=val_dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_perm[h] @ att_src[0, h]
                    alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                # Calculate attention weights.
                e = np.empty((heads, num_entries), dtype=val_dtype)
                softmax_sum = np.zeros((N, heads), dtype=val_dtype)
                C_vals = np.empty((heads, num_entries), dtype=dace.bool)

                for h, i in dace.map[0:heads, 0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    C_vals[h, i] = e_tmp > 0
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

                # SDDMM
                d_alpha_vals = np.empty((heads, num_entries), dtype=val_dtype)

                for h, i in dace.map[0:heads, 0:num_entries]:
                    d_alpha_vals[h, i] = 0

                # max_grid_size = int(1 << 31)
                # num_entries_round = 0
                # if num_entries > max_grid_size:
                #     num_entries_round = max_grid_size * (num_entries // max_grid_size)
                #     for seq in dace.map[0:num_entries_round:max_grid_size]:
                #         # for inner_i, h, k in dace.map[seq:seq+max_grid_size, 0:heads, 0:F_out]:
                #         for h, k, inner_i in dace.map[0:heads, 0:F_out, seq:seq+max_grid_size]:
                #             col = columns[inner_i]
                #             row = rows[inner_i]
                #             d_alpha_vals[inner_i, h] += output_grad_heads[col, h, k] * features[row, h, k]

                for h, k, inner_i in dace.map[0:heads, 0:F_out, 0:num_entries]:
                    col = columns[inner_i]
                    row = rows[inner_i]
                    d_alpha_vals[h, inner_i] += output_grad_heads[col, h, k] * features[row, h, k]

                dot_prods = np.zeros((N, heads), dtype=val_dtype)
                for h, i in dace.map[0:heads, 0:num_entries]:
                    col = columns[i]
                    dot_prods[col, h] += e[h, i] * d_alpha_vals[h, i]

                dl = np.zeros((heads, N), dtype=val_dtype)
                dr = np.zeros((heads, N), dtype=val_dtype)
                neg_slope = dace.define_local_scalar(val_dtype)
                neg_slope[:] = negative_slope
                for h, i in dace.map[0:heads, 0:num_entries]:
                    col = columns[i]
                    row = rows[i]
                    dE_val = dace.define_local_scalar(val_dtype)
                    dE_val[:] = (d_alpha_vals[h, i] - dot_prods[col, h]) * e[h, i]

                    dC_val = dace.define_local_scalar(val_dtype)
                    # dC_val[:] = dE_val * (C_vals[h, i] > 0) + dE_val * (
                    #         C_vals[h, i] <= 0) * neg_slope
                    dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * C_vals[h, i])
                    dr[h, row] += dC_val
                    dl[h, col] += dC_val

                dH_prime_perm = np.zeros((heads, N, F_out), dtype=val_dtype)

                output_grad_perm = np.transpose(output_grad_heads, (1, 0, 2))
                for h in range(heads):
                    coomm(A_rows=rows, A_cols=columns, A_vals=e[h], B=output_grad_perm[h],
                          C=dH_prime_perm[h],
                          beta=1.0, transA=False)

                dH_prime = np.empty((N, heads, F_out), dtype=val_dtype)
                for h, i, k in dace.map[0:heads, 0:N,
                               0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    dH_prime[i, h, k] = dH_prime_perm[h, i, k] + dl[h, i] * att_dst[0, h, k] + dr[
                        h, i] * att_src[0, h, k]

                out_reshaped_dH_prime[:] = np.reshape(dH_prime, (N, heads * F_out))
                # This has to be np.einsum('nf,nm->fm', out_reshaped_dH_prime, node_features),
                # not np.einsum('nm,nf->fm', node_features, out_reshaped_dH_prime), because
                # the latter is bugged for f = m.
                lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', out_reshaped_dH_prime,
                                                     node_features)

                # att_src_grad[:] = 0
                # att_dst_grad[:] = 0
                # for h, k, n in dace.map[0:heads, 0:F_out, 0:N]:
                #     att_dst_grad[0, h, k] += dl[n, h] * features[n, h, k]
                #     att_src_grad[0, h, k] += dr[n, h] * features[n, h, k]

                for h in dace.map[0:heads]:
                    att_src_grad[0, h, :] = dr[h] @ features_perm[h]  # N times N x F_out = F_out
                    # att_src_grad[0, h, :] = np.einsum('nf,n->f', features_perm[h], dr[h]) # h x n x f_out times h x n = h x f_out

                    att_dst_grad[0, h, :] = dl[h] @ features_perm[h]
                # att_dst_grad[:] = np.einsum('nhf,nh->hf', features, dl)  # F_out

                # for h, k in dace.map[0:heads, 0:F_out]:
                #     att_dst_grad[0, h, k] = 0
                #     for n in dace.map[0:N]:
                #         att_dst_grad[0, h, k] += dl[n, h] * features[n, h, k]
                #
                # for h, k in dace.map[0:heads, 0:F_out]:
                #     att_src_grad[0, h, k] = 0
                #     for n in dace.map[0:N]:
                #         att_src_grad[0, h, k] += dr[n, h] * features[n, h, k]
                bias_grad[:] = np.reshape(np.sum(output_grad, axis=0), (heads * F_out,))

        if compute_grad_for_node_features:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad,
                            node_features_grad):
                dH_prime_reshaped = np.empty((N, heads * F_out), dtype=val_dtype)
                basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                                   att_src, att_dst, att_src_grad, att_dst_grad,
                                   lin_srcDOTweight_grad, bias_grad, output_grad,
                                   dH_prime_reshaped)
                node_features_grad[:] = dH_prime_reshaped @ lin_srcDOTweight  # N x F_in
        else:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad):
                dH_prime_reshaped = np.empty((N, heads * F_out), dtype=val_dtype)
                basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                                   att_src, att_dst, att_src_grad, att_dst_grad,
                                   lin_srcDOTweight_grad, bias_grad, output_grad,
                                   dH_prime_reshaped)

        result_node, result = autodiff_utils.backward_program_for_node(
            backward_fn, context, forward_node)

        return result_node, result


@autoregister_params(op="torch_geometric.nn.conv.gat_conv.GATConv",
                     name="coo_cached")
class GATConvBackwardCOOCached(BackwardImplementation):
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
        one_min_neg_slope = 1 - negative_slope
        heads = forward_node.module.heads
        F_out = forward_node.module.out_channels

        val_dtype = node_features_desc.dtype
        compute_grad_for_node_features = 'node_features' in required_gradients

        @dace.program
        def basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                               att_src, att_dst, att_src_grad, att_dst_grad,
                               lin_srcDOTweight_grad, bias_grad, output_grad,
                               out_reshaped_dH_prime, e, is_pos_C_vals):
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

                dl = np.zeros((N,), dtype=val_dtype)
                dr = np.zeros((N,), dtype=val_dtype)
                neg_slope = dace.define_local_scalar(val_dtype)
                neg_slope[:] = negative_slope
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    row = rows[i]
                    dE_val = dace.define_local_scalar(val_dtype)
                    dE_val[:] = (d_alpha_vals[i] - dot_prods[col]) * e[i]

                    dC_val = dace.define_local_scalar(val_dtype)
                    # dC_val[:] = dE_val * (C_vals[i] > 0) + dE_val * (
                    #         C_vals[i] <= 0) * neg_slope
                    dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * is_pos_C_vals[i])
                    dr[row] += dC_val
                    dl[col] += dC_val

                out_reshaped_dH_prime[:] = np.zeros((N, F_out), dtype=val_dtype)

                coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=out_reshaped_dH_prime,
                      beta=1.0,
                      transA=False)

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    out_reshaped_dH_prime[i, k] += dl[i] * att_dst[0, 0, k]

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    out_reshaped_dH_prime[i, k] += dr[i] * att_src[0, 0, k]

                lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', out_reshaped_dH_prime,
                                                     node_features)  # F_out x F_in
                att_dst_grad[:] = dl @ features  # F_out
                att_src_grad[:] = dr @ features  # F_out
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

                features_perm = np.transpose(features, (1, 0, 2))

                ### COMPUTE THE GRADIENTS ###
                output_grad_heads = np.reshape(output_grad, (N, heads, F_out))

                # SDDMM
                d_alpha_vals = np.empty((num_entries, heads), dtype=val_dtype)

                for h, i in dace.map[0:heads, 0:num_entries]:
                    d_alpha_vals[i, h] = 0

                for h, k, inner_i in dace.map[0:heads, 0:F_out, 0:num_entries]:
                    col = columns[inner_i]
                    row = rows[inner_i]
                    d_alpha_vals[inner_i, h] += output_grad_heads[col, h, k] * features[row, h, k]

                dot_prods = np.zeros((N, heads), dtype=val_dtype)
                for h, i in dace.map[0:heads, 0:num_entries]:
                    col = columns[i]
                    dot_prods[col, h] += e[h, i] * d_alpha_vals[i, h]

                dl = np.zeros((heads, N), dtype=val_dtype)
                dr = np.zeros((heads, N), dtype=val_dtype)
                neg_slope = dace.define_local_scalar(val_dtype)
                neg_slope[:] = negative_slope
                for h, i in dace.map[0:heads, 0:num_entries]:
                    col = columns[i]
                    row = rows[i]
                    dE_val = dace.define_local_scalar(val_dtype)
                    dE_val[:] = (d_alpha_vals[i, h] - dot_prods[col, h]) * e[h, i]

                    dC_val = dace.define_local_scalar(val_dtype)
                    dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * is_pos_C_vals[h, i])
                    dr[h, row] += dC_val
                    dl[h, col] += dC_val

                dH_prime_perm = np.zeros((heads, N, F_out), dtype=val_dtype)

                output_grad_perm = np.transpose(output_grad_heads, (1, 0, 2))
                for h in range(heads):
                    coomm(A_rows=rows, A_cols=columns, A_vals=e[h], B=output_grad_perm[h],
                          C=dH_prime_perm[h],
                          beta=1.0, transA=False)

                dH_prime = np.empty((N, heads, F_out), dtype=val_dtype)
                for h, i, k in dace.map[0:heads, 0:N,
                               0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    dH_prime[i, h, k] = dH_prime_perm[h, i, k] + dl[h, i] * att_dst[0, h, k] + dr[
                        h, i] * att_src[0, h, k]

                out_reshaped_dH_prime[:] = np.reshape(dH_prime, (N, heads * F_out))
                # This has to be np.einsum('nf,nm->fm', out_reshaped_dH_prime, node_features),
                # not np.einsum('nm,nf->fm', node_features, out_reshaped_dH_prime), because
                # the latter is bugged for f = m.
                lin_srcDOTweight_grad[:] = np.einsum('nf,nm->fm', out_reshaped_dH_prime,
                                                     node_features)

                for h in dace.map[0:heads]:
                    att_src_grad[0, h, :] = dr[h] @ features_perm[h]  # N times N x F_out = F_out
                    att_dst_grad[0, h, :] = dl[h] @ features_perm[h]
                bias_grad[:] = np.reshape(np.sum(output_grad, axis=0), (heads * F_out,))

        if compute_grad_for_node_features:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad,
                            node_features_grad, e, is_pos_C_vals):
                dH_prime_reshaped = np.empty((N, heads * F_out), dtype=val_dtype)
                basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                                   att_src, att_dst, att_src_grad, att_dst_grad,
                                   lin_srcDOTweight_grad, bias_grad, output_grad,
                                   dH_prime_reshaped, e, is_pos_C_vals)
                node_features_grad[:] = dH_prime_reshaped @ lin_srcDOTweight  # N x F_in
        else:
            def backward_fn(node_features, rows, columns, lin_srcDOTweight,
                            att_src, att_dst, att_src_grad, att_dst_grad,
                            lin_srcDOTweight_grad, bias_grad, output_grad, e, is_pos_C_vals):
                dH_prime_reshaped = np.empty((N, heads * F_out), dtype=val_dtype)
                basic_gat_backward(node_features, rows, columns, lin_srcDOTweight,
                                   att_src, att_dst, att_src_grad, att_dst_grad,
                                   lin_srcDOTweight_grad, bias_grad, output_grad,
                                   dH_prime_reshaped, e, is_pos_C_vals)

        result_node, result = autodiff_utils.backward_program_for_node(
            backward_fn, context, forward_node)

        connect_output_from_forward(forward_node, result_node, context, 'e')
        connect_output_from_forward(forward_node, result_node, context, 'is_pos_C_vals')

        return result_node, result
