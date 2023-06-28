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

        N, F_out = output_shape
        node_features_desc = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features")
        F_in = node_features_desc.shape[1]

        row_shape = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "rows").shape
        num_entries = row_shape[0]

        negative_slope = forward_node.module.negative_slope
        heads = forward_node.module.heads

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
                # Uses H_prime (features), e (edge_vals), C_vals (C_vals).
                d_alpha_vals = np.zeros((num_entries,), dtype=val_dtype)
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    row = rows[i]
                    for k in dace.map[0:F_out]:
                        d_alpha_vals[i] += output_grad[col, k] * features[row, k]

                dot_prods = np.zeros((N,), dtype=val_dtype)
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    dot_prods[col] += e[i] * d_alpha_vals[i]

                dE_vals = np.zeros((num_entries,), dtype=val_dtype)
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    mult = (d_alpha_vals[i] - dot_prods[col]) * e[i]
                    dE_vals[i] = mult

                dC_vals = dE_vals * (C_vals > 0) + dE_vals * (C_vals <= 0) * negative_slope

                dl = np.zeros((N,), dtype=val_dtype)
                dr = np.zeros((N,), dtype=val_dtype)
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    dl[col] += dC_vals[i]

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    dr[row] += dC_vals[i]

                dH_prime = np.zeros((N, F_out), dtype=val_dtype)
                dH_prime[:] = 0
                # dH_prime += dl[:, None] @ att_dst[0, 0, None, :] + dr[:, None] @ att_src[0, 0, None, :]

                coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=dH_prime, beta=0.0,
                      transA=False)

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    dH_prime[i, k] += dl[i] * att_dst[0, 0, k]

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    dH_prime[i, k] += dr[i] * att_src[0, 0, k]

                weight_grad = np.zeros((F_out, F_in), dtype=val_dtype)
                weight_grad[:] = dH_prime.T @ node_features
                lin_srcDOTweight_grad[:] = weight_grad  # F_out x F_in
                # d_x = dH_prime @ lin_srcDOTweight  # N x F_in
                att_dst_grad[:] = features.T @ dl  # F_out
                att_src_grad[:] = features.T @ dr  # F_out
                bias_grad[:] = np.sum(output_grad, axis=0)
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
                    for k in dace.map[0:F_out]:
                        d_alpha_vals[i] += output_grad[col, k], features[row, k]

                dot_prods = np.zeros((N,), dtype=val_dtype)
                for i in range(num_entries):
                    col = columns[i]
                    dot_prods[col] += e[i] * d_alpha_vals[i]

                dE_vals = np.zeros((num_entries,), dtype=val_dtype)
                for i in range(num_entries):
                    col = columns[i]
                    dE_vals[i] = (d_alpha_vals[i] - dot_prods[col]) * e[i]

                dC_vals = dE_vals * (C_vals > 0) + dE_vals * (C_vals <= 0) * negative_slope

                dl = np.zeros((N,), dtype=val_dtype)
                dr = np.zeros((N,), dtype=val_dtype)
                for i in dace.map[0:num_entries]:
                    col = columns[i]
                    dl[col] += dC_vals[i]

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    dr[row] += dC_vals[i]

                dH_prime = np.zeros((N, F_out), dtype=val_dtype)
                dH_prime[:] = 0
                # dH_prime += dl[:, None] @ att_dst[0, 0, None, :] + dr[:, None] @ att_src[0, 0, None, :]

                coomm(A_rows=rows, A_cols=columns, A_vals=e, B=output_grad, C=dH_prime, beta=0.0,
                      transA=False)

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    mult = dl[i] * att_dst[0, 0, k]
                    dH_prime[i, k] += mult

                for i, k in dace.map[0:N, 0:F_out] @ dace.dtypes.ScheduleType.Sequential:
                    mult = dr[i] * att_src[0, 0, k]
                    dH_prime[i, k] += mult

                weight_grad = np.zeros((F_out, F_in), dtype=val_dtype)
                weight_grad[:] = dH_prime.T @ node_features
                lin_srcDOTweight_grad[:] = weight_grad  # F_out x F_in
                node_features_grad[:] = dH_prime @ lin_srcDOTweight  # N x F_in
                att_dst_grad[:] = features.T @ dl  # F_out
                att_src_grad[:] = features.T @ dr  # F_out
                bias_grad[:] = np.sum(output_grad, axis=0)


        result_node, result = autodiff_utils.backward_program_for_node(
            backward_fn, context, forward_node)

        return result_node, result
