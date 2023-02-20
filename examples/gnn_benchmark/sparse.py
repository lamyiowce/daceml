from typing import Optional, Tuple

import torch
import torch_geometric
import torch_sparse


class GraphMatrix:
    def to_input_list(self):
        raise NotImplementedError


class CsrGraph(GraphMatrix, torch_sparse.SparseTensor):
    def __init__(self,
                 node_features: torch.Tensor,
                 *args, **kwargs):
        self.node_features = node_features
        super().__init__(*args, **kwargs)

    def to_input_list(self):
        edge_rowptr, edge_col, edge_weights = self.csr()
        if edge_weights is not None:
            return self.node_features, edge_rowptr, edge_col, edge_weights
        else:
            return self.node_features, edge_rowptr, edge_col

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        sparse_matrix = cls(node_features=data.x,
                                 row=edge_index[0],
                                 rowptr=None,
                                 col=edge_index[1],
                                 value=edge_weight)

        return sparse_matrix


class CooGraph(GraphMatrix, torch_sparse.SparseTensor):
    def __init__(self,
                 node_features: torch.Tensor,
                 *args, **kwargs):
        self.node_features = node_features
        super().__init__(*args, **kwargs)

    def to_input_list(self):
        edge_row, edge_col, edge_weights = self.coo()
        if edge_weights is not None:
            return self.node_features, edge_row, edge_col, edge_weights
        else:
            return self.node_features, edge_row, edge_col

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        sparse_matrix = cls(node_features=data.x,
                                 row=edge_index[0],
                                 rowptr=None,
                                 col=edge_index[1],
                                 value=edge_weight)
        return sparse_matrix
