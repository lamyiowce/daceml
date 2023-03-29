from typing import Optional, List, Union, Tuple

import dace
import numpy as np
import torch
from dace import nodes as nd
from dace.registry import autoregister_params
from dace.transformation.interstate import StateFusion
from torch_sparse import SparseTensor

import daceml.autodiff.utils as autodiff_utils
from daceml.autodiff import BackwardImplementation, BackwardContext, BackwardResult
from daceml.torch.module import DaceModule
from examples.gnn_benchmark.models import GCN
from examples.gnn_benchmark.sdfg_util import set_implementation

N = 4
num_entries = 9
num_out_features = 3
num_in_features = 6

np.random.seed(2137)
torch.random.manual_seed(2137)


@autoregister_params(op="MaxPool", name="default")
class GCNConvBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        output_shape = autodiff_utils.forward_out_desc_with_name(
            forward_node, context, "Y").shape

        N, C, H, W = output_shape
        sty, stx = forward_node.strides
        sy, sx = forward_node.kernel_shape

        def gcn_backward(X, Y_grad, X_grad):
            for b, c, ti, tj in dace.map[0:N, 0:C, 0:H, 0:W]:
                maxv = np.empty([1], dtype=dace.float32)
                maxi = np.empty([1], dtype=dace.int32)
                maxj = np.empty([1], dtype=dace.int32)
                with dace.tasklet:
                    v >> maxv
                    v = -9999999

                # Deterministic argmax (assuming sequential map)
                for i, j in dace.map[0:sy, 0:sx]:
                    with dace.tasklet:
                        o << X[b, c, sty * ti + i, stx * tj + j]
                        vin << maxv
                        v >> maxv(-1)
                        ind_i >> maxi(-1)
                        ind_j >> maxj(-1)
                        if o > vin:
                            v = o
                            ind_i = i
                            ind_j = j
                with dace.tasklet:
                    igrad << Y_grad[b, c, ti, tj]
                    ind_i << maxi
                    ind_j << maxj
                    ograd >> X_grad(1)[b, c, :, :]
                    ograd[ind_i, ind_j] = igrad

        result_node, result = autodiff_utils.backward_program_for_node(
            maxpool_backward, context, forward_node)

        return result_node, result

def torch_prog(node_features, adj_matrix, weight, bias):
    return (adj_matrix.T @ (node_features @ weight.T) + bias).sum()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features = torch.randn(N, num_in_features, dtype=torch.float32).to(device)

    adj_matrix = torch.tensor([[1., 0, 1, 0],
                               [1., 1, 1, 0],
                               [0., 1, 1, 1],
                               [0., 0, 1, 0]]).to(device)
    adj = SparseTensor.from_dense(adj_matrix)
    rowptrs, columns, edge_vals = adj.csr()
    weights = torch.randn(num_out_features, num_in_features, dtype=torch.float32).to(device)
    bias = torch.randn(num_out_features, dtype=torch.float32).to(device)
    total = torch.zeros((1,), dtype=torch.float32).to(device)

    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    torch_model = GCN(num_node_features=num_in_features, num_classes=num_out_features, num_hidden_features=10,
                      normalize=False)
    dace_model = DaceModule(torch_model, backward=False, inputs_to_skip=['1', '2', '3'])
    dace_model.prepend_post_onnx_hook('set impl', lambda m: set_implementation(m, 'csr'))

    def fuse_states(m):
        m.sdfg.apply_transformations_repeated(StateFusion, permissive=True, progress=True)
    dace_model.append_post_onnx_hook("state_fusion", fuse_states)

    criterion = torch.nn.NLLLoss()
    prediction = dace_model(node_features, rowptrs, columns, edge_vals)
    loss = criterion(prediction, y)
    print(f"gradients before: {dace_model.model.conv1.lin.weight.grad}")

    # gradients can flow through model!
    loss.backward()

    print(f"gradients after: {dace_model.model.conv1.lin.weight.grad}")

    torch_loss = criterion(torch_model(node_features), y)
    torch_loss.backward()
    print(f"gradient torch: {torch_model.conv1.lin.weight.grad}")

    check_equal(dace_model.model.fc3.weight.grad, torch_model.fc3.weight.grad)
    # torch_total = torch_prog(node_features, adj_matrix, weights, bias)
    #
    # check_equal(total, torch_total)


def check_equal(result, expected_result):
    if np.allclose(result, expected_result):
        print("\n==== Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print("\n*↯*↯*↯* INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ *↯*↯*↯*")
    print("Actual output:")
    print(result)
    print("Expected output:")
    print(expected_result)


if __name__ == '__main__':
    main()
