import torch
from torch_sparse import SparseTensor


def make_torch_edge_list_args(data, add_edge_weights, input_grad: bool):
    '''Create an argument list for the torch edge list model.'''
    input_features = torch.clone(data.x, memory_format=torch.contiguous_format).detach()
    input_features.requires_grad = input_grad
    torch_edge_list_args = input_features, data.edge_index.contiguous()
    if add_edge_weights:
        torch_edge_list_args += (data.edge_weight.contiguous(),)
    return torch_edge_list_args


def make_torch_csr_args(data, input_grad: bool):
    """Create argument lists for torch CSR models."""
    num_nodes = data.num_nodes
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight, sparse_sizes=(num_nodes, num_nodes))

    # pyg requires the sparse tensor input to be transposed.
    input_features = torch.clone(data.x, memory_format=torch.contiguous_format).detach()
    input_features.requires_grad = input_grad
    torch_csr_args = input_features, sparse_edge_index.t()
    return torch_csr_args
