from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torch_geometric
import torch_sparse


class GraphMatrix:
    def to_input_list(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def parse_args(args_list) -> Dict[str, Union[str, int, float]]:
        raise NotImplementedError


class SparseGraph(GraphMatrix):
    @staticmethod
    def parse_args(args_list):
        if len(args_list) > 0:
            raise ValueError(f'Unexpected arguments: {args_list}')
        return {}

    def __init__(self,
                 node_features: torch.Tensor,
                 compute_input_grad: bool,
                 *args, **kwargs
                 ):
        if node_features is not None:
            self.node_features = node_features.contiguous()
            self.node_features.requires_grad = compute_input_grad
        else:
            self.node_features = None

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, compute_input_grad: bool,
                      idx_dtype=None, **kwargs):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        sparse_matrix = cls(node_features=data.x,
                            edge_vals=edge_weight,
                            rows=edge_index[0],
                            cols=edge_index[1],
                            idx_dtype=idx_dtype,
                            compute_input_grad=compute_input_grad,
                            **kwargs)
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
                 idx_dtype: Optional[torch.dtype] = None,
                 *args,
                 **kwargs
                 ):
        idx_dtype = idx_dtype or rows.dtype
        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()
        self.edge_vals = torch.clone(edge_vals,
                                     memory_format=torch.contiguous_format) if edge_vals is not None else None
        self.rowptrs = torch.clone(rowptrs.to(idx_dtype), memory_format=torch.contiguous_format)
        self.columns = torch.clone(columns.to(idx_dtype), memory_format=torch.contiguous_format)
        del sparse
        super().__init__(node_features, *args, **kwargs)

    def data_list(self):
        data_list = self.rowptrs, self.columns
        return data_list + (
            self.edge_vals,) if self.edge_vals is not None else data_list

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}('
        if self.node_features is not None:
            repr_str += f'node_features={self.node_features.shape},'
        if self.edge_vals is not None:
            repr_str += f' edge_vals={self.edge_vals.shape},'
        repr_str += f' rows={self.rowptrs.shape}, cols={self.columns.shape})'
        return repr_str


class CooGraph(SparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: Optional[torch.Tensor],
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype: Optional[torch.dtype] = None,
                 *args,
                 **kwargs
                 ):
        idx_dtype = idx_dtype or rows.dtype
        if edge_vals is not None:
            self.edge_vals = edge_vals.contiguous()
        else:
            self.edge_vals = None
        self.cols = cols.to(idx_dtype).contiguous()
        self.rows = rows.to(idx_dtype).contiguous()
        super().__init__(node_features, *args, **kwargs)

    def data_list(self):
        data_list = self.rows, self.cols
        return data_list + (
            self.edge_vals,) if self.edge_vals is not None else data_list

    def __repr__(self):
        repr_str = f'CooGraph('
        if self.node_features is not None:
            repr_str += f'node_features={self.node_features.shape},'
        if self.edge_vals is not None:
            repr_str += f' edge_vals={self.edge_vals.shape},'
        repr_str += f' rows={self.rows.shape}, cols={self.cols.shape})'
        return repr_str


class CscGraph(SparseGraph):

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 idx_dtype: Optional[torch.dtype] = None,
                 *args,
                 **kwargs,
                 ):
        idx_dtype = idx_dtype or rows.dtype
        N = node_features.shape[0]
        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols,
                                           sparse_sizes=(N, N))
        colptrs, rows, edge_vals = sparse.csc()
        if edge_vals is not None:
            self.edge_vals = torch.clone(edge_vals, memory_format=torch.contiguous_format)
        else:
            self.edge_vals = None
        self.colptrs = torch.clone(colptrs.to(idx_dtype), memory_format=torch.contiguous_format)
        self.rows = torch.clone(rows.to(idx_dtype), memory_format=torch.contiguous_format)
        del sparse
        super().__init__(node_features, *args, **kwargs)

    def data_list(self):
        if self.edge_vals is not None:
            return self.colptrs, self.rows, self.edge_vals
        else:
            return self.colptrs, self.rows


class HybridCsrCooGraph(SparseGraph):
    @staticmethod
    def parse_args(args):
        if 0 < len(args) <= 1:
            return {"csr_cutoff": float(args[0])}
        else:
            return {}

    def __init__(self,
                 node_features: Optional[torch.Tensor],
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 csr_cutoff: float,
                 idx_dtype: Optional[torch.dtype] = None,
                 *args,
                 **kwargs,
                 ):
        idx_dtype = idx_dtype or rows.dtype

        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        cols, rows = None, None
        rowptrs, columns, edge_vals = sparse.csr()
        row_lens = rowptrs[1:] - rowptrs[:-1]
        cutoff_len = int(torch.quantile(row_lens.to(float), csr_cutoff))
        self.cutoff_len = cutoff_len

        csr_row_lens = torch.minimum(row_lens, torch.tensor(cutoff_len))

        # `coo_row_idx` has -1 if the entry goes to csr, otherwise it has the row index
        # of the coo entry.
        coo_row_idx = -1 * torch.ones_like(columns)
        for row_idx, (row_start, csr_row_len, row_end) in enumerate(
                zip(rowptrs[:-1], csr_row_lens, rowptrs[1:])):
            coo_row_idx[row_start + csr_row_len:row_end] = row_idx
        csr_rowptrs = torch.zeros_like(rowptrs, dtype=idx_dtype)
        csr_rowptrs[1:] = torch.cumsum(csr_row_lens, dim=0)
        self.csr_rowptrs = csr_rowptrs
        self.csr_cols = columns[coo_row_idx == -1].to(idx_dtype).contiguous()

        self.coo_cols = columns[coo_row_idx >= 0].to(idx_dtype).contiguous()
        self.coo_rows = coo_row_idx[coo_row_idx >= 0].to(idx_dtype).contiguous()

        if edge_vals is not None:
            self.csr_edge_vals = edge_vals[coo_row_idx == -1].contiguous()
            self.coo_edge_vals = edge_vals[coo_row_idx >= 0].contiguous()
        else:
            self.csr_edge_vals = None
            self.coo_edge_vals = None

        assert (self.csr_cols.shape[0] + self.coo_cols.shape[0] ==
                columns.shape[
                    0]), 'CSR and COO columns do not add up to the total number of columns.'

        super().__init__(node_features, *args, **kwargs)

    def data_list(self):
        if self.csr_edge_vals is not None and self.coo_edge_vals is not None:
            return self.csr_rowptrs, self.csr_cols, self.csr_edge_vals, self.coo_rows, self.coo_cols, self.coo_edge_vals
        else:
            return self.csr_rowptrs, self.csr_cols, self.coo_rows, self.coo_cols

    def __repr__(self):
        repr = f'HybridCsrCooGraph(cutoff_len={self.cutoff_len}, '
        if self.node_features is not None:
            repr += f'node_features={self.node_features.shape}, '
        if self.csr_edge_vals is not None:
            repr += f'csr_edge_vals={self.csr_edge_vals.shape}, '
        if self.coo_edge_vals is not None:
            repr += f'coo_edge_vals={self.coo_edge_vals.shape}, '
        return repr + f'csr_cols={self.csr_cols.shape}, coo_cols={self.coo_cols.shape},' \
                      f' csr_rowptrs={self.csr_rowptrs.shape}, coo_rows={self.coo_rows.shape})'


class HybridCscCooGraph(SparseGraph):
    @staticmethod
    def parse_args(args):
        if 0 < len(args) <= 1:
            return {"csr_cutoff": float(args[0])}
        else:
            return {}

    def __init__(self,
                 node_features: Optional[torch.Tensor],
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 csr_cutoff: float,
                 idx_dtype: Optional[torch.dtype] = None,
                 *args,
                 **kwargs,
                 ):
        # Construct a HybridCsrCooGraph with the transposed data.
        csr_coo = HybridCsrCooGraph(node_features, edge_vals, rows=cols,
                                    cols=rows, csr_cutoff=csr_cutoff,
                                    idx_dtype=idx_dtype, *args, **kwargs)
        self.cutoff_len = csr_coo.cutoff_len
        # CSC is transposed CSR.
        self.csc_colptrs = csr_coo.csr_rowptrs
        self.csc_rows = csr_coo.csr_cols
        self.csc_edge_vals = csr_coo.csr_edge_vals
        self.coo_rows = csr_coo.coo_cols
        self.coo_cols = csr_coo.coo_rows
        self.coo_edge_vals = csr_coo.coo_edge_vals
        super().__init__(node_features, *args, **kwargs)

    def data_list(self):
        if self.csc_edge_vals is not None and self.coo_edge_vals is not None:
            return self.csc_colptrs, self.csc_rows, self.csc_edge_vals, self.coo_rows, self.coo_cols, self.coo_edge_vals
        else:
            return self.csc_colptrs, self.csc_rows, self.coo_rows, self.coo_cols

    def __repr__(self):
        repr = f'HybridCsrCooGraph(cutoff_len={self.cutoff_len}, '
        if self.node_features is not None:
            repr += f'node_features={self.node_features.shape}, '
        if self.csc_edge_vals is not None:
            repr += f'csr_edge_vals={self.csc_edge_vals.shape}, '
        if self.coo_edge_vals is not None:
            repr += f'coo_edge_vals={self.coo_edge_vals.shape}, '
        return repr + f'csc_rows={self.csc_rows.shape}, coo_cols={self.coo_cols.shape},' \
                      f' csc_colptrs={self.csc_colptrs.shape}, coo_rows={self.coo_rows.shape})'


class EllpackGraph(GraphMatrix):
    @staticmethod
    def parse_args(args):
        if len(args) != 1:
            raise ValueError(
                f'EllpackGraph requires exactly one argument: block size.')
        return {"block_size": int(args[0])}

    def __init__(self,
                 node_features: torch.Tensor,
                 edge_vals: torch.Tensor,
                 rows: torch.Tensor,
                 cols: torch.Tensor,
                 block_size: int,
                 compute_input_grad: bool,
                 idx_dtype: Optional[torch.dtype] = None,
                 ):
        if node_features is not None:
            self.node_features = node_features.contiguous()
            self.node_features.requires_grad = compute_input_grad
        else:
            self.node_features = None

        idx_dtype = idx_dtype or rows.dtype
        device = rows.device

        sparse = torch_sparse.SparseTensor(value=edge_vals, row=rows, col=cols)
        rowptrs, columns, edge_vals = sparse.csr()
        num_nodes = rowptrs.shape[0] - 1
        if num_nodes % block_size != 0:
            raise ValueError(
                f"Ellpack block size ({block_size} should divide the number of nodes ({num_nodes}).")

        num_blocked_rows = num_nodes // block_size
        col_block_idxs = columns // block_size

        ell_columns = -torch.ones((num_blocked_rows, num_blocked_rows),
                                  dtype=columns.dtype,
                                  device=device)
        max_num_blocks_in_row = 0
        for i, (a, b) in enumerate(
                zip(rowptrs[:-block_size:block_size],
                    rowptrs[block_size::block_size])):
            unique_idxs = col_block_idxs[a:b].unique()
            unique_idxs.sort()
            ell_columns[i, :unique_idxs.shape[0]] = unique_idxs
            num_blocks_in_row = unique_idxs.shape[0]
            max_num_blocks_in_row = max(max_num_blocks_in_row,
                                        num_blocks_in_row)

        self.columns = ell_columns[:, :max_num_blocks_in_row].to(
            idx_dtype).contiguous()

        self.values = torch.zeros(
            (num_nodes, max_num_blocks_in_row * block_size),
            dtype=edge_vals.dtype, device=device)
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
                target_indices = row_cols[
                                     this_block_selector] % block_size + j * block_size
                # Place the values in the value matrix.
                self.values[i, target_indices] = row_vals[this_block_selector]

        self.num_zero_vals = (self.values == 0).to(torch.float32).mean()

        block_stats = torch.zeros((block_size * block_size + 1,),
                                  dtype=torch.int64)
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
                      compute_input_grad: bool,
                      idx_dtype: Union[str, torch.dtype] = 'keep'):
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[0],
                   cols=data.edge_index[1],
                   idx_dtype=idx_dtype, block_size=block_size,
                   compute_input_grad=compute_input_grad)

    @classmethod
    def from_dense(cls, adjacency_matrix: torch.Tensor,
                   node_features: Optional[torch.Tensor],
                   block_size: int,
                   idx_dtype: Union[str, torch.dtype] = 'keep'):
        csr_matrix = torch_sparse.SparseTensor.from_dense(adjacency_matrix)
        rows, col, val = csr_matrix.coo()
        return EllpackGraph(node_features, rows=rows, cols=col, edge_vals=val,
                            block_size=block_size,
                            idx_dtype=idx_dtype)


class EllpackTransposedGraph(EllpackGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = self.columns

    def to_input_list(self):
        return self.node_features, self.rows, self.values

    @classmethod
    def from_pyg_data(cls, data: torch_geometric.data.Data, block_size: int,
                      compute_input_grad: bool,
                      idx_dtype: Union[str, torch.dtype] = 'keep'):
        # Same as for EllpackGraph, but the rows and cols are switched.
        return cls(data.x, edge_vals=data.edge_weight, rows=data.edge_index[1],
                   cols=data.edge_index[0],
                   idx_dtype=idx_dtype, block_size=block_size,
                   compute_input_grad=compute_input_grad)
