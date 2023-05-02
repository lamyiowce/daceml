from typing import List, Optional, Tuple, Union

import dace
import numpy as np
from dace.registry import autoregister_params
from dace.sdfg import nodes as nd

from daceml.autodiff import BackwardImplementation, BackwardContext, \
    BackwardResult, utils as autodiff_utils
from examples.gnn_benchmark import csrmm_libnode
from examples.gnn_benchmark.sparse_mm.coomm import coomm


@autoregister_params(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                     name="csr")
class GCNConvBackward(BackwardImplementation):
    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:
        output_shape = autodiff_utils.forward_out_desc_with_name(
            forward_node, context, "output").shape

        N, F = output_shape
        node_features_desc = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features")
        M = node_features_desc.shape[1]
        val_dtype = node_features_desc.dtype


        compute_grad_for_node_features = 'node_features' in required_gradients

        def gcn_backward(node_features, rowptrs, columns, edge_vals,
                         linDOTweight,
                         linDOTweight_grad, bias_grad,
                         output_grad):
            """
            node_features: input features, N x M
            rowptrs: row pointers (CSR format), N+1
            columns: col, num_entries
            edge_vals: values, num_entries
            linDOTweight: F x M
            bias: F
            output: N x F

            node_features_grad: N x M
            linDOTweight_grad: F x M
            output_grad: N x F
            """

            # Compute the values of the gradient of weights and node_features.
            # The gradient of the bias is just the sum of the output gradient.
            # The gradient of the adjacency matrix is not computed.

            # Compute the gradient of the GCN layer.
            # Grad W = Grad C^T @ A^t @ X
            temp = dace.define_local((N, M), dtype=val_dtype)
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, node_features,
                                temp, transA=True)
            linDOTweight_grad[:] = np.einsum('ji,jk->ik', output_grad, temp)

            bias_grad[:] = np.sum(output_grad, axis=0)

        def gcn_backward_with_node_features(node_features, rowptrs, columns,
                                            edge_vals,
                                            linDOTweight, node_features_grad,
                                            linDOTweight_grad, bias_grad,
                                            output_grad):
            # Grad X = A @ Grad G @ W
            temp = dace.define_local((N, M), dtype=val_dtype)
            temp[:] = output_grad @ linDOTweight
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, temp,
                                node_features_grad)
            gcn_backward(node_features, rowptrs, columns, edge_vals,
                         linDOTweight,
                         linDOTweight_grad, bias_grad, output_grad)

        result_node, result = autodiff_utils.backward_program_for_node(
            gcn_backward_with_node_features if compute_grad_for_node_features else gcn_backward,
            context, forward_node)

        return result_node, result


@autoregister_params(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                     name="csr_reorder")
class GCNConvBackward(BackwardImplementation):
    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:
        output_shape = autodiff_utils.forward_out_desc_with_name(
            forward_node, context, "output").shape

        N, F = output_shape
        node_features_desc = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features")
        M = node_features_desc.shape[1]
        val_dtype = node_features_desc.dtype


        compute_grad_for_node_features = 'node_features' in required_gradients

        def gcn_backward(node_features, rowptrs, columns, edge_vals,
                         linDOTweight,
                         linDOTweight_grad, bias_grad,
                         output_grad):
            """
            node_features: input features, N x M
            rowptrs: row pointers (CSR format), N+1
            columns: col, num_entries
            edge_vals: values, num_entries
            linDOTweight: F x M
            bias: F
            output: N x F

            node_features_grad: N x M
            linDOTweight_grad: F x M
            output_grad: N x F
            """

            # Compute the values of the gradient of weights and node_features.
            # The gradient of the bias is just the sum of the output gradient.
            # The gradient of the adjacency matrix is not computed.

            # Compute the gradient of the GCN layer.
            # Grad W = Grad C^T @ A^t @ X
            temp = dace.define_local((N, M), dtype=val_dtype)
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, node_features,
                                temp, transA=True)
            linDOTweight_grad[:] = np.einsum('ji,jk->ik', output_grad, temp)
            bias_grad[:] = np.sum(output_grad, axis=0)

        def gcn_backward_with_node_features(node_features, rowptrs, columns,
                                            edge_vals,
                                            linDOTweight, node_features_grad,
                                            linDOTweight_grad, bias_grad,
                                            output_grad):
            # Grad X = A @ Grad G @ W
            temp = dace.define_local((N, F), dtype=val_dtype)
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, output_grad,
                                temp)
            node_features_grad[:] = temp @ linDOTweight
            gcn_backward(node_features, rowptrs, columns, edge_vals,
                         linDOTweight,
                         linDOTweight_grad, bias_grad, output_grad)

        result_node, result = autodiff_utils.backward_program_for_node(
            gcn_backward_with_node_features if compute_grad_for_node_features else gcn_backward,
            context, forward_node)

        return result_node, result

@autoregister_params(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                     name="coo")
class GCNConvBackward(BackwardImplementation):
    @staticmethod
    def backward(
            forward_node: nd.Node, context: BackwardContext,
            given_gradients: List[Optional[str]],
            required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:
        output_shape = autodiff_utils.forward_out_desc_with_name(
            forward_node, context, "output").shape

        N, F = output_shape
        node_features_desc = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features")
        M = node_features_desc.shape[1]
        edge_vals_shape = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "edge_vals").shape
        K = edge_vals_shape[0]

        val_dtype = node_features_desc.dtype
        compute_grad_for_node_features = 'node_features' in required_gradients

        def gcn_backward(node_features, rows, columns, edge_vals,
                         linDOTweight_grad, bias_grad, output_grad):
            """
            node_features: input features, N x M
            rowptrs: row pointers (CSR format), N+1
            columns: col, K
            edge_vals: values, K
            linDOTweight: F x M
            bias: F
            output: N x F

            node_features_grad: N x M
            linDOTweight_grad: F x M
            output_grad: N x F
            """

            # Compute the values of the gradient of weights and node_features.
            # The gradient of the bias is just the sum of the output gradient.
            # The gradient of the adjacency matrix is not computed.

            # Compute the gradient of the GCN layer.
            # Grad W = Grad C^T @ A^t @ X
            temp = dace.define_local((N, M), dtype=val_dtype)
            coomm(rows, columns, edge_vals, node_features, temp, transA=True,
                  beta=0.0)
            linDOTweight_grad[:] = np.einsum('ji,jk->ik', output_grad, temp)

            bias_grad[:] = np.sum(output_grad, axis=0)

        def gcn_backward_with_node_features(node_features, rows, columns,
                                            edge_vals,
                                            linDOTweight, node_features_grad,
                                            linDOTweight_grad, bias_grad,
                                            output_grad):
            # Grad X = A @ Grad G @ W
            temp = dace.define_local((N, M), dtype=val_dtype)
            temp[:] = output_grad @ linDOTweight

            gcn_backward(node_features, rows, columns, edge_vals,
                         linDOTweight_grad, bias_grad, output_grad)

            # `columns` and `rows` are switched because transA=False doesn't work here
            # with CuSPARSE for some reason. It seems to be a bug in CuSPARSE?
            coomm(columns, rows, edge_vals, temp, node_features_grad, beta=0.0,
                  transA=True)

        result_node, result = autodiff_utils.backward_program_for_node(
            gcn_backward_with_node_features if compute_grad_for_node_features else gcn_backward,
            context, forward_node)

        return result_node, result
