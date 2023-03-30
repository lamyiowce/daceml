import abc
from typing import Type, Dict, Union

import dace
import numpy as np
from dace import nodes, SDFG, SDFGState

from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation
from daceml.onnx.op_implementations.utils import program_for_node
from daceml.util.utils import in_desc_with_name
from examples.gnn_benchmark import csrmm_libnode, sparse
from examples.gnn_benchmark.implementations.common import SparseLayerBase


class GCNConvBase(SparseLayerBase, metaclass=abc.ABCMeta):
    """
    A GCN node, given node features X, weights W and adjacency matrix A,
    computes: X' = A.t @ (X @ W.t)
    """

    @classmethod
    def forward(cls, node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nodes.Node, SDFG]:
        if node.module.add_self_loops:
            raise NotImplementedError("Adding self loops is not supported. "
                                      "Add self-loops in preprocessing.")
        if node.module.normalize:
            raise NotImplementedError(
                "Normalization is not implemented. "
                "Normalize edge weights in preprocessing.")

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

        gcn_op = cls.make_op(N, num_out_features, num_entries, dtype, do_bias)

        return program_for_node(gcn_op, sdfg, state, node)


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="semester_thesis")
class GCNConvSemesterThesis(GCNConvBase):
    graph_format: sparse.GraphMatrix = sparse.CsrGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'rowptrs': dace.int64,
        'columns': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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
    graph_format: sparse.GraphMatrix = sparse.CsrGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'rowptrs': dace.int64,
        'columns': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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
                """
                features = dace.define_local((N, num_out_features), dtype=dtype)
                features[:] = np.einsum('ij,kj->ik', node_features,
                                        linDOTweight)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] = bias[j]
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


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv", name="csc")
class GCNConvCSC(GCNConvBase):
    graph_format: sparse.GraphMatrix = sparse.CscGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'colptrs': dace.int64,
        'rows': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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
    graph_format: sparse.GraphMatrix = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'rows': dace.int64,
        'columns': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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
            features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)

            output[:] = 0
            for i, k in dace.map[0:num_entries, 0:num_out_features]:
                c = columns[i]
                r = rows[i]
                output[c, k] += edge_vals[i] * features[r, k]

        if do_bias:
            def bias_prog(node_features, rows, columns, edge_vals,
                          linDOTweight, bias, output):
                gcn_op(node_features, rows, columns, edge_vals,
                       linDOTweight, output)
                for i, j in dace.map[0:N, 0:num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gcn_op


@op_implementation(op="torch_geometric.nn.conv.gcn_conv.GCNConv",
                   name="ellpack_t")
class GCNConvEllpackTransposed(GCNConvBase):
    graph_format: sparse.GraphMatrix = sparse.EllpackTransposedGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'rows': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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

            for i, k in dace.map[0:N, 0:num_out_features]:
                for j in dace.map[0:num_entries]:
                    row = rows[i, j]
                    mult = edge_vals[i, j] * features[row, k]
                    output[i, k] += mult

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
                   name="ellpack")
class GCNConvEllpack(GCNConvBase):
    graph_format: sparse.GraphMatrix = sparse.EllpackGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': dace.float32,
        'columns': dace.int64,
        'edge_vals': dace.float32,
    }

    @staticmethod
    def make_op(N: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, do_bias: bool):
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
