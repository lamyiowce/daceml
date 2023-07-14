import copy

import numpy as np
import torch
from torch_sparse import SparseTensor

from daceml.torch.module import DaceModule
from examples.gnn_benchmark.models import GCN
from examples.gnn_benchmark.sdfg_util import set_implementation
import examples.gnn_benchmark.implementations.gcn_backward

# Mark import as used. This contains the backward implementation.
assert examples.gnn_benchmark.implementations.gcn_backward
N = 4
num_entries = 9
num_out_features = 5
num_classes = 3
num_in_features = 6

np.random.seed(2137)
torch.random.manual_seed(2137)


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

    criterion = torch.nn.CrossEntropyLoss()
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
        print(f"==== {tag}Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print(f"*↯*↯*↯* {tag}INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ *↯*↯*↯*")
        print("Actual output:")
        print(result)
        print("Expected output:")
        print(expected_result)


if __name__ == '__main__':
    main()
