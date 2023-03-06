from typing import Optional

import torch
import torch_geometric
import torch_sparse


class GraphMatrix:
    def to_input_list(self):
        raise NotImplementedError


class TorchSparseGraph(GraphMatrix, torch_sparse.SparseTensor):
    def __init__(self,
                 node_features: torch.Tensor,
                 *args, **kwargs):
        self.node_features = node_features
        super().__init__(*args, **kwargs)

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


class CsrGraph(TorchSparseGraph):
    def to_input_list(self):
        edge_rowptr, edge_col, edge_weights = self.csr()
        if edge_weights is not None:
            return self.node_features, edge_rowptr, edge_col, edge_weights
        else:
            return self.node_features, edge_rowptr, edge_col


class CooGraph(TorchSparseGraph):
    def to_input_list(self):
        edge_row, edge_col, edge_weights = self.coo()
        if edge_weights is not None:
            return self.node_features, edge_row, edge_col, edge_weights
        else:
            return self.node_features, edge_row, edge_col


class CscGraph(TorchSparseGraph):
    def to_input_list(self):
        edge_colptr, edge_row, edge_weights = self.csc()
        if edge_weights is not None:
            return self.node_features, edge_colptr, edge_row, edge_weights
        else:
            return self.node_features, edge_colptr, edge_row


class EllpackGraph(GraphMatrix):
    def __init__(self, node_features: torch.Tensor, rowptrs: torch.Tensor,
                 columns: torch.Tensor, vals: Optional[torch.Tensor]):
        self.node_features = node_features
        device = node_features.device
        num_rows = rowptrs.shape[0] - 1
        max_elems_in_row = torch.max(rowptrs[1:] - rowptrs[:-1]).item()
        if vals is not None:
            self.vals = torch.zeros(num_rows, max_elems_in_row,
                                    dtype=torch.float32).to(device)
        self.columns = torch.zeros(num_rows, max_elems_in_row,
                                   dtype=torch.int64).to(device)

        for i in range(num_rows):
            len = rowptrs[i + 1] - rowptrs[i]
            self.columns[i, :len] = columns[rowptrs[i]:rowptrs[i + 1]]
            if vals is not None:
                self.vals[i, :len] = vals[rowptrs[i]:rowptrs[i + 1]]

    def to_input_list(self):
        return self.node_features, self.columns, self.vals

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        sparse_matrix = torch_sparse.SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_weight)
        rowptr, col, val = sparse_matrix.csr()
        return cls(data.x, rowptr, col, val)
