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
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        self.node_features = node_features
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype

        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()

        device = rowptrs.device
        num_rows = rowptrs.shape[0] - 1
        max_elems_in_row = torch.max(rowptrs[1:] - rowptrs[:-1]).item()
        if edge_vals is not None:
            self.vals = torch.zeros(num_rows, max_elems_in_row,
                                    dtype=edge_vals.dtype).to(device)
        self.columns = torch.zeros(num_rows, max_elems_in_row,
                                   dtype=columns.dtype).to(device)

        for i in range(num_rows):
            len = rowptrs[i + 1] - rowptrs[i]
            self.columns[i, :len] = columns[rowptrs[i]:rowptrs[i + 1]]
            if edge_vals is not None:
                self.vals[i, :len] = edge_vals[rowptrs[i]:rowptrs[i + 1]]

        self.columns = self.columns.to(idx_dtype)

    def to_input_list(self):
        return self.node_features, self.columns, self.vals

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, idx_dtype: Union[str, torch.dtype] = 'keep'):
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[0], cols=data.edge_index[1],
                   idx_dtype=idx_dtype)

    @classmethod
    def from_dense(cls, adjacency_matrix: torch.Tensor, node_features: Optional[torch.Tensor]):
        csr_matrix = torch_sparse.SparseTensor.from_dense(adjacency_matrix)
        rows, col, val = csr_matrix.coo()
        return EllpackGraph(node_features, rows=rows, cols=col, edge_vals=val)


class EllpackTransposedGraph(EllpackGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = self.columns

    def to_input_list(self):
        return self.node_features, self.rows, self.vals

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, idx_dtype: Union[str, torch.dtype] = 'keep'):
        # Same as for EllpackGraph, but the rows and cols are switched.
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[1], cols=data.edge_index[0],
                   idx_dtype=idx_dtype)
