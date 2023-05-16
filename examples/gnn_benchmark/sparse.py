from typing import Optional, Union, List

import torch
import torch_geometric
import torch_sparse


class GraphMatrix:
    def to_input_list(self) -> List[torch.Tensor]:
        raise NotImplementedError


class SparseGraph(GraphMatrix):

    def __init__(self,
                 node_features: torch.Tensor,
                 *args, **kwargs
                 ):
        self.node_features = node_features.contiguous()

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


class CsrGraph(SparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: Optional[torch.Tensor],
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()
        self.edge_vals = edge_vals.contiguous() if edge_vals is not None else None
        self.rowptrs = rowptrs.to(idx_dtype).contiguous()
        self.columns = columns.to(idx_dtype).contiguous()
        super().__init__(node_features)

    def data_list(self):
        data_list = self.rowptrs, self.columns
        return data_list + (self.edge_vals,) if self.edge_vals is not None else data_list


class CooGraph(SparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: Optional[torch.Tensor],
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype='keep'
                 ):
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        if edge_vals is not None:
            self.edge_vals = edge_vals.contiguous()
        else:
            self.edge_vals = None
        self.cols = cols.to(idx_dtype).contiguous()
        self.rows = rows.to(idx_dtype).contiguous()
        super().__init__(node_features)

    def data_list(self):
        data_list = self.rows, self.cols
        return data_list + (self.edge_vals,) if self.edge_vals is not None else data_list


class CscGraph(SparseGraph):

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
        self.edge_vals = edge_vals.contiguous()
        self.colptrs = colptrs.to(idx_dtype).contiguous()
        self.rows = rows.to(idx_dtype).contiguous()
        super().__init__(node_features)

    def data_list(self):
        return self.colptrs, self.rows, self.edge_vals


class EllpackGraph(GraphMatrix):
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 block_size: int,
                 idx_dtype='keep'
                 ):
        self.node_features = node_features.contiguous()
        idx_dtype = rows.dtype if idx_dtype == 'keep' else idx_dtype
        device = rows.device

        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()
        num_nodes = rowptrs.shape[0] - 1
        if num_nodes % block_size != 0:
            raise ValueError(f"Ellpack block size ({block_size} should divide the number of nodes ({num_nodes}).")

        num_blocked_rows = num_nodes // block_size
        col_block_idxs = columns // block_size

        ell_columns = -torch.ones((num_blocked_rows, num_blocked_rows), dtype=columns.dtype, device=device)
        max_num_blocks_in_row = 0
        for i, (a, b) in enumerate(zip(rowptrs[:-block_size:block_size], rowptrs[block_size::block_size])):
            unique_idxs = col_block_idxs[a:b].unique()
            unique_idxs.sort()
            ell_columns[i, :unique_idxs.shape[0]] = unique_idxs
            num_blocks_in_row = unique_idxs.shape[0]
            max_num_blocks_in_row = max(max_num_blocks_in_row, num_blocks_in_row)

        self.columns = ell_columns[:, :max_num_blocks_in_row].to(idx_dtype).contiguous()

        self.values = torch.zeros((num_nodes, max_num_blocks_in_row * block_size), dtype=edge_vals.dtype, device=device)
        for i, (a, b) in enumerate(zip(rowptrs[:-1], rowptrs[1:])):
            row_cols = columns[a:b]
            row_col_block_idxs = col_block_idxs[a:b]
            row_vals = edge_vals[a:b]

            for j in range(max_num_blocks_in_row):
                # Get the column index of the block.
                col_idx = self.columns[i // block_size, j]
                # Find all entries that should end up in this block.
                this_block_selector = row_col_block_idxs == col_idx
                # Compute indices where the entries should be placed in the value matrix.
                target_indices = row_cols[this_block_selector] % block_size + j * block_size
                # Place the values in the value matrix.
                self.values[i, target_indices] = row_vals[this_block_selector]

        self.num_zero_vals = (self.values == 0).to(torch.float32).mean()

        block_stats = torch.zeros((block_size * block_size + 1,), dtype=torch.int64)
        for i in range(0, max_num_blocks_in_row * block_size, block_size):
            for j in range(0, num_nodes, block_size):
                patch = self.values[i:i + block_size, j:j + block_size]
                num_nonzero = (patch != 0).sum()
                block_stats[num_nonzero] += 1

        self.block_stats = block_stats
        print("Num zero vals:", self.num_zero_vals)
        print("Block stats:", block_stats)

    def to_input_list(self):
        return self.node_features, self.columns, self.values

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, block_size: int,
                      idx_dtype: Union[str, torch.dtype] = 'keep'):
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[0], cols=data.edge_index[1],
                   idx_dtype=idx_dtype, block_size=block_size)

    @classmethod
    def from_dense(cls, adjacency_matrix: torch.Tensor, node_features: Optional[torch.Tensor], block_size: int, idx_dtype: Union[str, torch.dtype] = 'keep'):
        csr_matrix = torch_sparse.SparseTensor.from_dense(adjacency_matrix)
        rows, col, val = csr_matrix.coo()
        return EllpackGraph(node_features, rows=rows, cols=col, edge_vals=val, block_size=block_size, idx_dtype=idx_dtype)


class EllpackTransposedGraph(EllpackGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = self.columns

    def to_input_list(self):
        return self.node_features, self.rows, self.values

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, block_size: int, idx_dtype: Union[str, torch.dtype] = 'keep'):
        # Same as for EllpackGraph, but the rows and cols are switched.
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[1], cols=data.edge_index[0],
                   idx_dtype=idx_dtype, block_size=block_size)
