import copy
from typing import Optional, List, Union, Tuple

import dace
import numpy as np
import torch
from dace import nodes as nd
from dace.registry import autoregister_params
from torch_sparse import SparseTensor

import daceml.autodiff.utils as autodiff_utils
from daceml.autodiff import BackwardImplementation, BackwardContext, \
    BackwardResult
from daceml.torch.module import DaceModule
from examples.gnn_benchmark import csrmm_libnode
from examples.gnn_benchmark.models import GCN
from examples.gnn_benchmark.sdfg_util import set_implementation

N = 4
num_entries = 9
num_out_features = 5
num_classes = 3
num_in_features = 6

np.random.seed(2137)
torch.random.manual_seed(2137)


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
            for i, k in dace.map[0:N, 0:M]:
                for j in dace.map[rowptrs[i]:rowptrs[i + 1]]:
                    # Below lines result in compile errors when enabling thread block dynamic scheduling.
                    column = columns[j]
                    mult = node_features[i, k] * edge_vals[j]
                    temp[column, k] += mult

            linDOTweight_grad[:] = output_grad.T @ temp

            node_features_grad[:] = 0
            temp[:] = output_grad @ linDOTweight
            csrmm_libnode.csrmm(rowptrs, columns, edge_vals, temp,
                                node_features_grad)
            bias_grad[:] = np.sum(output_grad, axis=0)

        result_node, result = autodiff_utils.backward_program_for_node(
            gcn_backward, context, forward_node)

        return result_node, result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features = torch.randn(N, num_in_features, dtype=torch.float32).to(
        device)

    adj_matrix = torch.tensor([[1., 0, 1, 0],
                               [1., 1, 1, 0],
                               [0., 1, 1, 1],
                               [0., 0, 1, 0]]).to(device)
    adj = SparseTensor.from_dense(adj_matrix)
    rowptrs, columns, edge_vals = adj.csr()

    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    torch_model = GCN(num_node_features=num_in_features,
                      num_classes=num_classes,
                      num_hidden_features=num_out_features,
                      normalize=False)
    torch_model.train()
    dace_model = DaceModule(copy.deepcopy(torch_model), backward=True,
                            inputs_to_skip=['1', '2', '3'])
    dace_model.prepend_post_onnx_hook('set impl',
                                      lambda m: set_implementation(m, 'csr'))

    criterion = torch.nn.NLLLoss()
    dace_prediction = dace_model(node_features, rowptrs, columns, edge_vals)
    loss = criterion(dace_prediction, y)
    print(f"gradients before: {dace_model.model.conv1.lin.weight.grad}")

    # gradients can flow through model!
    loss.backward()

    torch_prediction = torch_model(node_features, adj.t())
    torch_loss = criterion(torch_prediction, y)
    torch_loss.backward()

    check_equal(dace_prediction, torch_prediction)
    dace_parameters = dict(dace_model.model.named_parameters())
    for name, parameter in torch_model.named_parameters():
        check_equal(dace_parameters[name], parameter, name=name)
        check_equal(dace_parameters[name].grad, parameter.grad,
                    name=name + ".grad")


def check_equal(result, expected_result, name=None):
    tag = (name + ': ') if name else ''
    if np.allclose(result.detach(), expected_result.detach()):
        print(f"\n==== {tag}Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print(f"\n*↯*↯*↯* {tag}INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ *↯*↯*↯*")
        print("Actual output:")
        print(result)
        print("Expected output:")
        print(expected_result)


if __name__ == '__main__':
    main()
