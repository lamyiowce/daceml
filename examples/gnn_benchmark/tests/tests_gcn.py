import dace
import numpy as np
import pytest
import torch
import torch_geometric.data
from torch import nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from daceml.torch.module import DaceModule
from examples.gnn_benchmark.util import register_replacement_overrides, name_to_impl_class

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_implementation(dace_module, implementation):
    def hook(dace_module):
        for node, _ in dace_module.sdfg.all_nodes_recursive():
            if (isinstance(node, dace.sdfg.nodes.LibraryNode)
                    and implementation in node.implementations):
                node.implementation = implementation

    dace_module.prepend_post_onnx_hook("set_implementation", hook)


@pytest.mark.parametrize("bias", [False, True], ids=['', 'bias'])
@pytest.mark.parametrize("implementation", ['csr', 'semester_thesis', 'csr_coo'])
def test_gcn(bias, implementation):
    self_loops = False
    normalize = False

    weights_values = torch.Tensor([[1, 1], [0, 0], [1, 0]])
    bias_values = torch.Tensor([0.21, 0.37, 0])

    register_replacement_overrides(implementation_name=implementation, layer_name='gcn',
                                   idx_dtype=torch.int64,
                                   val_dtype=torch.float32)

    sdfg_name = f'GCN_{implementation}_{self_loops}_{normalize}_{bias}'

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(2,
                                 3,
                                 bias=bias,
                                 normalize=normalize,
                                 add_self_loops=self_loops)
            self.conv1.lin.weight = nn.Parameter(weights_values)
            if bias:
                self.conv1.bias = nn.Parameter(bias_values)

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            return x

    model = DaceModule(GCN(), sdfg_name=sdfg_name)
    set_implementation(model, implementation)

    edges = torch.tensor([[0, 0, 0, 0, 2, 2, 3, 4], [0, 1, 2, 4, 0, 2, 4, 4]], dtype=torch.int64)
    edge_values = torch.tensor([1., 2., 3., 4., 5., 6, 7, 8])
    x = torch.tensor([[0., 1], [1, 1], [-1, 0], [0, -1], [1, 0]])
    pyg_data = torch_geometric.data.Data(x=x, edge_index=edges, edge_weight=edge_values)
    inputs = name_to_impl_class['gcn'][implementation].convert_data(pyg_data).to_input_list()

    pred = model(*inputs)

    original_gcnconv = GCN()
    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = original_gcnconv(x, edges, edge_values).detach().numpy()

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)


def test_simple():
    weights_values = torch.Tensor([[1, 1], [0, 0], [1, 0]])

    edges = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]])
    edge_values = torch.tensor([1., 2., 3., 4., 5.])
    adj_matrix = SparseTensor.from_edge_index(edges, edge_attr=edge_values)
    dense = adj_matrix.to_dense()
    x = torch.tensor([[0., 1], [1, 1], [-1, 0]])
    original_gcnconv = GCNConv(3, 2, normalize=False, add_self_loops=False, bias=False)
    original_gcnconv.lin.weight = nn.Parameter(weights_values)
    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = original_gcnconv(x, adj_matrix.t()).detach().numpy()

    pred = dense.T @ (x @ weights_values.T)

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)
