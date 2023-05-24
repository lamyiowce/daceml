import dace
import numpy as np
import pytest
import scipy
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from daceml.torch.module import DaceModule
from examples.gnn_benchmark.util import register_replacement_overrides

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_implementation(dace_module, implementation):
    def hook(dace_module):
        for node, _ in dace_module.sdfg.all_nodes_recursive():
            if (isinstance(node, dace.sdfg.nodes.LibraryNode)
                    and implementation in node.implementations):
                node.implementation = implementation

    dace_module.prepend_post_onnx_hook("set_implementation", hook)


@pytest.mark.parametrize("bias", [True], ids=['bias'])
@pytest.mark.parametrize("implementation", ['csr'])
@pytest.mark.parametrize("N,F,heads", [(2, 3, 2), (2, 3, 1), (120, 20, 8)])
def test_gat(bias, implementation, N, F, heads):
    F_in = F
    F_out = F + 3
    torch.random.manual_seed(42)
    # weights_values = torch.rand((F_out, F_in))
    # bias_values = torch.rand((F_out,))

    register_replacement_overrides(implementation_name=implementation,
                                   layer_name='gat', idx_dtype=torch.int64,
                                   val_dtype=torch.float32)

    sdfg_name = f'GAT_{implementation}_{bias}'

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(F_in,
                                 F_out,
                                 bias=bias,
                                 heads=heads,
                                 add_self_loops=False)

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            return x

    reference_model = GAT()
    model = DaceModule(reference_model, sdfg_name=sdfg_name)
    set_implementation(model, implementation)

    # Create input.
    edges = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]])
    edge_values = torch.tensor([1., 2., 3., 4., 5.])
    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A))
    rowptr, col, _ = adj_matrix.csr()
    x = torch.rand((N, F_in))

    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = reference_model(x, adj_matrix.t()).detach().numpy()

    pred = model(x, rowptr, col)

    print('\nCalculated: \n', pred)
    print('Expected: \n', expected_pred)
    assert np.allclose(pred, expected_pred)

