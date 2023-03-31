from typing import List, Optional, Tuple, Union

import dace
import numpy as np
from dace.registry import autoregister_params
from dace.sdfg import nodes as nd

from daceml.autodiff import BackwardImplementation, BackwardContext, \
    BackwardResult, utils as autodiff_utils
from examples.gnn_benchmark import csrmm_libnode


@autoregister_params(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                     name="default")
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
        node_features_shape = autodiff_utils.forward_in_desc_with_name(
            forward_node, context, "node_features").shape
        M = node_features_shape[1]

        def gcn_backward(node_features, rowptrs, columns, edge_vals,
                         linDOTweight, bias, node_features_grad,
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
            temp = np.zeros((N, M), dtype=dace.float32)
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, node_features,
                                temp, transA=True)

            # Grad W = Grad C^T @ A^t @ X
            linDOTweight_grad[:] = output_grad.T @ temp

            # Grad X = A @ Grad G @ W
            node_features_grad[:] = 0
            temp[:] = output_grad @ linDOTweight
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, temp,
                                node_features_grad)
            bias_grad[:] = np.sum(output_grad, axis=0)

        result_node, result = autodiff_utils.backward_program_for_node(
            gcn_backward, context, forward_node)

        return result_node, result
