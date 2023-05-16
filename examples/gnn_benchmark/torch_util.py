from torch_sparse import SparseTensor


def make_torch_edge_list_args(data, add_edge_weights):
    '''Create an argument list for the torch edge list model.'''
    torch_edge_list_args = data.x.contiguous(), data.edge_index.contiguous()
    if add_edge_weights:
        torch_edge_list_args += (data.edge_weight.contiguous(),)
    return torch_edge_list_args


def make_torch_csr_args(data):
    """Create argument lists for torch CSR models."""
    num_nodes = data.num_nodes
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight, sparse_sizes=(num_nodes, num_nodes))

    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = data.x.contiguous(), sparse_edge_index.t()
    return torch_csr_args
