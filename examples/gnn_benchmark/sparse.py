from typing import Optional, Union

import scipy.sparse
import torch
import torch_geometric
import torch_sparse


class GraphMatrix:
    def to_input_list(self):
        raise NotImplementedError


class ScipySparseGraph(GraphMatrix):

    def __init__(self,
                 node_features: torch.Tensor,
                 *args, **kwargs
                 ):
        self.node_features = node_features

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, idx_dtype='keep'):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        sparse_matrix = cls(node_features=data.x,
                            edge_vals=edge_weight,
                            rows=edge_index[0],
                            cols=edge_index[1],
                            idx_dtype=idx_dtype)
        return sparse_matrix

    def to_input_list(self):
        input_list = (self.node_features,) + self.data_list()
        if input_list[-1] is None:
            return input_list[:-1]
        return input_list


class CsrGraph(ScipySparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()
        self.edge_vals = edge_vals
        self.rowptrs = rowptrs.to(idx_dtype)
        self.columns = columns.to(idx_dtype)
        super().__init__(node_features)

    def data_list(self):
        return self.rowptrs, self.columns, self.edge_vals


class CooGraph(ScipySparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        self.edge_vals = edge_vals
        self.cols = cols.to(idx_dtype)
        self.rows = rows.to(idx_dtype)
        super().__init__(node_features)

    def data_list(self):
        return self.rows, self.cols, self.edge_vals


class CscGraph(ScipySparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        colptrs, rows, edge_vals = sparse.csc()
        self.edge_vals = edge_vals
        self.colptrs = colptrs.to(idx_dtype)
        self.rows = rows.to(idx_dtype)
        super().__init__(node_features)

    def data_list(self):
        return self.colptrs, self.rows, self.edge_vals

class EllpackGraph(GraphMatrix):
    def __init__(self, node_features: Optional[torch.Tensor], rowptrs: torch.Tensor,
                 columns: torch.Tensor, vals: Optional[torch.Tensor]):
        self.node_features = node_features
        device = rowptrs.device
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
    def from_pyg_data(cls, data: torch_geometric.data.Data, idx_dtype: Union[str, torch.dtype] = 'keep'):
        idx_dtype = data.edge_index.dtype if idx_dtype == 'keep' else idx_dtype
        edge_index = data.edge_index.to(idx_dtype)
        edge_weight = data.edge_weight

        sparse_matrix = torch_sparse.SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_weight)
        rowptr, col, val = sparse_matrix.csr()
        return cls(data.x, rowptr, col, val)

    @classmethod
    def from_dense(cls, adjacency_matrix: torch.Tensor, node_features: Optional[torch.Tensor]):
        csr_matrix = torch_sparse.SparseTensor.from_dense(adjacency_matrix)
        rowptr, col, val = csr_matrix.csr()
        return EllpackGraph(node_features, rowptrs=rowptr, columns=col, vals=val)


class EllpackTransposedGraph(GraphMatrix):
    def __init__(self, node_features: torch.Tensor, colptrs: torch.Tensor,
                 rows: torch.Tensor, vals: Optional[torch.Tensor]):
        self.node_features = node_features
        ellpack = EllpackGraph(node_features, colptrs, rows, vals)
        _, self.rows, self.vals = ellpack.to_input_list()

    def to_input_list(self):
        return self.node_features, self.rows, self.vals

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, idx_dtype: Union[str, torch.dtype] = 'keep'):
        idx_dtype = data.edge_index.dtype if idx_dtype == 'keep' else idx_dtype
        edge_index = data.edge_index.to(idx_dtype)
        edge_weight = data.edge_weight

        sparse_matrix = torch_sparse.SparseTensor(
            row=edge_index[0],
            rowptr=None,
            col=edge_index[1],
            value=edge_weight)

        colptr, rows, val = sparse_matrix.csc()
        return cls(data.x, colptr, rows, val)
