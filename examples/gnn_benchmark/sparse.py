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

    def to_input_list(self):
        input_list = (self.node_features,) + self.data_list()
        if input_list[-1] is None:
            return input_list[:-1]
        return input_list


class CsrGraph(TorchSparseGraph):
    data_list = TorchSparseGraph.csr


class CooGraph(TorchSparseGraph):
    data_list = TorchSparseGraph.coo


class CscGraph(TorchSparseGraph):
    data_list = TorchSparseGraph.csc


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


class EllpackTransposedGraph(GraphMatrix):
    def __init__(self, node_features: torch.Tensor, colptrs: torch.Tensor,
                    rows: torch.Tensor, vals: Optional[torch.Tensor]):
        self.node_features = node_features
        ellpack = EllpackGraph(node_features, colptrs, rows, vals)
        _, self.rows, self.vals = ellpack.to_input_list()

    def to_input_list(self):
        return self.node_features, self.rows, self.vals

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        sparse_matrix = torch_sparse.SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_weight)

        colptr, rows, val = sparse_matrix.csc()
        return cls(data.x, colptr, rows, val)
