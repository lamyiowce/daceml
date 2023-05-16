import abc
from typing import Dict, Union, Callable

import dace
import numpy as np
import torch
import torch_geometric
from dace import nodes, SDFG, SDFGState

import examples.gnn_benchmark.implementations.gcn_backward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation
from daceml.onnx.op_implementations.utils import program_for_node
from daceml.util.utils import in_desc_with_name
from examples.gnn_benchmark import csrmm_libnode, sparse
from examples.gnn_benchmark.implementations import common
from examples.gnn_benchmark.implementations.common import SparseLayerBase
from examples.gnn_benchmark.sparse_mm.blocked_ellpack_mm import \
    blocked_ellpack_mm
from examples.gnn_benchmark.sparse_mm.coomm import coomm

# Mark this import as used. It's needed to register the backward pass.
assert examples.gnn_benchmark.implementations.gcn_backward


class GCNConvBase(SparseLayerBase, metaclass=abc.ABCMeta):
    """
    A GCN node, given node features X, weights W and adjacency matrix A,
    computes: X' = A.t @ X @ W.t
    """

    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.module.add_self_loops:
            raise NotImplementedError("Adding self loops is not supported. "
                                      "Add self-loops in preprocessing.")
        if node.module.normalize:
            raise NotImplementedError(
                "Normalization is not implemented. "
                "Normalize edge weights in preprocessing.")

        return True

    @classmethod
    def forward(cls, node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nodes.Node, SDFG]:
        N, do_bias, dtype, num_entries, num_in_features, num_out_features = cls.get_info(
            node, state, sdfg)

        gcn_op = cls.make_op(N, num_in_features, num_out_features, num_entries,
                             dtype, do_bias)

        return program_for_node(gcn_op, sdfg, state, node)

    @staticmethod
    def get_info(node: onnx_op.ONNXOp, state: SDFGState, sdfg: SDFG):
        features_desc = in_desc_with_name(node, state, sdfg, "node_features")
        N, num_in_features = features_desc.shape
        dtype = features_desc.dtype
        weights_desc = in_desc_with_name(node, state, sdfg, "linDOTweight")
        num_out_features = weights_desc.shape[0]
        try:
            col_desc = in_desc_with_name(node, state, sdfg, "columns")
            num_entries = col_desc.shape[-1]
        except ValueError:
            # In the CSC format there is no `columns` array, but the `rows`
            # array.
            row_desc = in_desc_with_name(node, state, sdfg, "rows")
            num_entries = row_desc.shape[-1]
        do_bias = 'bias' in [inp.name for inp in node.schema.inputs]
        return N, do_bias, dtype, num_entries, num_in_features, num_out_features


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="semester_thesis")
class GCNConvSemesterThesis(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CsrGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rowptrs': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        def gcn_op(node_features, rowptrs, columns, edge_vals,
                   linDOTweight, output):
            """
            node_features: input features, N x M
            rowptrs: row pointers (CSR format), N+1
            columns: col, num_entries
            edge_vals: values, num_entries
            linDOTweight: F x M
            output: N x F
            """
            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            output[:] = 0
            for i, k in dace.map[0:N, 0:num_out_features]:
                for j in dace.map[rowptrs[i]:rowptrs[i + 1]]:
                    # Below lines result in compile errors when enabling thread
                    # block dynamic scheduling.
                    column = columns[j]
                    mult = features[i, k] * edge_vals[j]
                    output[column, k] += mult

        if do_bias:
            def bias_prog(node_features, rowptrs, columns, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, rowptrs, columns, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv", name="csr")
class GCNConvCSR(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CsrGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rowptrs': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        if do_bias:
            def gcn_op(node_features, rowptrs, columns, edge_vals,
                       linDOTweight, bias, output):
                """
                node_features: input features, N x M
                rowptrs: row pointers (CSR format), N+1
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F

                Compute X' = A.t @ X @ W.t + b
                """

                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] = bias[j]
                features = dace.define_local((N, num_out_features), dtype=dtype)
                features[:] = np.einsum('ij,kj->ik', node_features,
                                        linDOTweight)
                csrmm_libnode.csrmm(rowptrs, columns, edge_vals, features,
                                    output, beta=1.0, transA=True)

        else:
            def gcn_op(node_features, rowptrs, columns, edge_vals,
                       linDOTweight, output):
                """
                node_features: input features, N x M
                rowptrs: row pointers (CSR format), N+1
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                features = dace.define_local((N, num_out_features), dtype=dtype)
                features[:] = np.einsum('ij,kj->ik', node_features,
                                        linDOTweight)
                csrmm_libnode.csrmm(rowptrs, columns, edge_vals, features,
                                    output, transA=True)

        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="csr_reorder")
class GCNConvCSRReordered(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CsrGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rowptrs': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        if do_bias:
            def gcn_op(node_features, rowptrs, columns, edge_vals,
                       linDOTweight, bias, output):
                """
                node_features: input features, N x M
                rowptrs: row pointers (CSR format), N+1
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F

                Compute X' = A.t @ X @ W.t + b
                """

                aggregated_features = dace.define_local((N, num_in_features),
                                                        dtype=dtype)
                csrmm_libnode.csrmm(rowptrs, columns, edge_vals, node_features,
                                    aggregated_features, beta=0.0, transA=True)
                output[:] = np.einsum('ij,kj->ik', aggregated_features,
                                      linDOTweight)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

        else:
            def gcn_op(node_features, rowptrs, columns, edge_vals,
                       linDOTweight, output):
                """
                node_features: input features, N x M
                rowptrs: row pointers (CSR format), N+1
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                aggregated_features = dace.define_local((N, num_in_features),
                                                        dtype=dtype)
                csrmm_libnode.csrmm(rowptrs, columns, edge_vals, node_features,
                                    aggregated_features, beta=0.0, transA=True)
                output[:] = np.einsum('ij,kj->ik', aggregated_features,
                                      linDOTweight)

        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv", name="csc")
class GCNConvCSC(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CscGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'colptrs': common.SpecialInputType.IDX_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        def gcn_op(node_features, colptrs, rows, edge_vals,
                   linDOTweight, output):
            """
            node_features: input features, N x M
            colptrs: row pointers (CSR format), N+1
            rows: col, num_entries
            edge_vals: values, num_entries
            linDOTweight: F x M
            output: N x F
            """
            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            # A is in CSC format, so in order to compute A.t @ B, we call CSR
            # matmul. This is because CSC(A) = CSR(A.t).
            csrmm_libnode.csrmm(colptrs, rows, edge_vals, features, output,
                                transA=False)

        if do_bias:
            def bias_prog(node_features, colptrs, rows, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, colptrs, rows, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv", name="coo")
class GCNConvCOO(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CooGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        if do_bias:
            def gcn_op(node_features, rows, columns, edge_vals,
                       linDOTweight, bias, output):
                """
                node_features: input features, N x M
                row: row idxs (COO format), num_entries
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                features = dace.define_local((N, num_out_features), dtype=dtype)
                features[:] = np.einsum('ij,kj->ik', node_features,
                                        linDOTweight)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, edge_vals, features, output, beta=1.0,
                      transA=True)
        else:
            def gcn_op(node_features, rows, columns, edge_vals,
                       linDOTweight, output):
                """
                node_features: input features, N x M
                row: row idxs (COO format), num_entries
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                features = dace.define_local((N, num_out_features), dtype=dtype)
                features[:] = np.einsum('ij,kj->ik', node_features,
                                        linDOTweight)
                coomm(rows, columns, edge_vals, features, output, transA=True)
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv", name="coo_adapt")
class GCNConvCOOAdapt(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.CooGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        if do_bias:
            # Y = A.t @ X @ W.t + b
            def gcn_op(node_features, rows, columns, edge_vals,
                       linDOTweight, bias, output):
                """
                node_features: input features, N x M
                row: row idxs (COO format), num_entries
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                if num_in_features > num_out_features:
                    # Y = A.t @ (X @ W.t) + b
                    features = dace.define_local((N, num_out_features), dtype=dtype)
                    features[:] = np.einsum('ij,kj->ik', node_features,
                                            linDOTweight)
                    for i, j in dace.map[0:N, 0:num_out_features]:
                        output[i, j] = bias[j]
                    coomm(rows, columns, edge_vals, features, output, beta=1.0,
                          transA=True)
                else:
                    # Y = (A.t @ X) @ W.t + b
                    temp = dace.define_local((N, num_in_features), dtype=dtype)
                    coomm(rows, columns, edge_vals, node_features, temp, beta=0.0,
                          transA=True)
                    output[:] = np.einsum('nm,fm->nf', temp,
                                            linDOTweight)
                    for i, j in dace.map[0:N, 0:num_out_features]:
                        output[i, j] += bias[j]
        else:
            def gcn_op(node_features, rows, columns, edge_vals,
                       linDOTweight, output):
                """
                node_features: input features, N x M
                row: row idxs (COO format), num_entries
                columns: col, num_entries
                edge_vals: values, num_entries
                linDOTweight: F x M
                output: N x F
                """
                if num_in_features > num_out_features:
                    features = dace.define_local((N, num_out_features), dtype=dtype)
                    features[:] = np.einsum('ij,kj->ik', node_features,
                                            linDOTweight)
                    coomm(rows, columns, edge_vals, features, output, beta=1.0,
                          transA=True)
                else:
                    temp = dace.define_local((N, num_out_features), dtype=dtype)
                    coomm(rows, columns, edge_vals, node_features, temp, beta=0.0,
                          transA=True)
                    output[:] = np.einsum('ij,kj->ik', temp,
                                            linDOTweight)
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="ellpack_t")
class GCNConvEllpackTransposed(GCNConvBase):
    convert_data: Callable[
        [
            torch_geometric.data.Data], sparse.GraphMatrix] = sparse.EllpackTransposedGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        # num_entries is the maximal number of values in a column.
        def gcn_op(node_features, rows, edge_vals,
                   linDOTweight, output):
            """
            node_features: input features, N x M
            rows: col, N x max_column_entries
            edge_vals: N x max_column_entries
            linDOTweight: F x M
            output: N x F
            """
            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            output[:] = 0

            blocked_ellpack_mm(A_ellcolind=rows, A_ellvalues=edge_vals,
                               B=features, C=output,
                               beta=0.0, transA=False)

        if do_bias:
            def bias_prog(node_features, rows, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, rows, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="ellpack_pure")
class GCNConvEllpack(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.EllpackGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                num_entries: int, dtype: dace.dtypes.Typeclasses,
                do_bias: bool):
        # num_entries is the maximal number of values in a row.
        def gcn_op(node_features, columns, edge_vals,
                   linDOTweight, output):
            """
            node_features: input features, N x M
            columns: col, N x max_row_entries
            edge_vals: N x max_row_entries
            linDOTweight: F x M
            output: N x F
            """
            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            output[:] = 0

            for i, k in dace.map[0:N, 0:num_out_features]:
                for j in dace.map[0:num_entries]:
                    # i: row idx.
                    column = columns[i, j]
                    mult = edge_vals[i, j] * features[i, k]
                    output[column, k] += mult

        if do_bias:
            def bias_prog(node_features, columns, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, columns, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="ellpack")
class GCNConvEllpack(GCNConvBase):
    convert_data: Callable[[
        torch_geometric.data.Data], sparse.GraphMatrix] = sparse.EllpackGraph.from_pyg_data
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
        'edge_vals': common.SpecialInputType.VAL_DTYPE,
    }
    allowed_idx_dtypes = [torch.int32]

    @classmethod
    def forward(cls, node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nodes.Node, SDFG]:
        N, do_bias, dtype, max_num_blocks_in_row, num_in_features, num_out_features = cls.get_info(
            node, state, sdfg)

        col_desc = in_desc_with_name(node, state, sdfg, "columns")
        num_column_rows = col_desc.shape[1]
        values_desc = in_desc_with_name(node, state, sdfg, "edge_vals")
        num_values_rows = values_desc.shape[1]
        block_size = num_values_rows // num_column_rows
        gcn_op = cls.make_op(N=N, num_in_features=num_in_features,
                             num_out_features=num_out_features,
                             dtype=dtype, do_bias=do_bias,
                             block_size=block_size)

        return program_for_node(gcn_op, sdfg, state, node)

    @staticmethod
    def make_op(N: int, num_in_features: int, num_out_features: int,
                dtype: dace.dtypes.Typeclasses,
                do_bias: bool, block_size: int):
        def gcn_op(node_features, columns, edge_vals,
                   linDOTweight, output):
            """
            node_features: input features, N x M
            columns: col, N x max_row_entries
            edge_vals: N x max_row_entries
            linDOTweight: F x M
            output: N x F
            """
            features = dace.define_local((N, num_out_features), dtype=dtype)
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            blocked_ellpack_mm(A_ellcolind=columns, A_ellvalues=edge_vals,
                               B=features, C=output,
                               beta=0.0, transA=True)

        if do_bias:
            def bias_prog(node_features, columns, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, columns, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op
