import abc
from typing import List, Union, Dict

import dace
import numpy as np
import onnx
from dace import nodes, SDFG, SDFGState
from onnx import helper

import examples.gnn_benchmark.implementations.gat_backward
from daceml.onnx import shape_infer_GATConv, ArraySpec, convert_attribute_proto
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation
from daceml.onnx.op_implementations.utils import program_for_node
from daceml.util.utils import in_desc_with_name
from examples.gnn_benchmark import sparse
from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.implementations import common
from examples.gnn_benchmark.implementations.common import SparseLayerBase, \
    SpecialInputType
from examples.gnn_benchmark.sparse_mm.coomm import coomm

# Mark this import as used. It's needed to register the backward pass.
assert examples.gnn_benchmark.implementations.gat_backward


class GATConvBase(SparseLayerBase, metaclass=abc.ABCMeta):
    ssi_fn = shape_infer_GATConv

    @classmethod
    def forward(cls, node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> Union[nodes.Node, SDFG]:
        if node.module.add_self_loops:
            raise NotImplementedError("Adding self loops is not supported. "
                                      "Add self-loops in preprocessing.")

        features_desc = in_desc_with_name(node, state, sdfg, "node_features")
        N, num_in_features = features_desc.shape
        dtype = features_desc.dtype

        try:
            col_desc = in_desc_with_name(node, state, sdfg, "columns")
            num_entries, = col_desc.shape
        except ValueError:
            rows_desc = in_desc_with_name(node, state, sdfg, "rows")
            num_entries, = rows_desc.shape

        heads = node.module.heads
        num_out_features = node.module.out_channels
        negative_slope = node.module.negative_slope
        assert negative_slope < 1.0

        do_bias = 'bias' in [inp.name for inp in node.schema.inputs]

        op = cls.make_op(N, heads, num_out_features, num_entries, dtype,
                         negative_slope, do_bias)

        return program_for_node(op, sdfg, state, node)


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="semester_thesis")
class GATConvSemesterThesis(GATConvBase):
    graph_format = sparse.CsrGraph
    input_spec = {
        "node_features": SpecialInputType.VAL_DTYPE,
        "rowptrs": SpecialInputType.IDX_DTYPE,
        "columns": SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                   att_src, att_dst, output):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F
            att_dst: H x F
            output: N x H * F'
            """

            # Transform input features.
            features = dace.define_local((N, heads, num_out_features),
                                         dtype=dtype)
            features_tmp = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
            features[:] = np.reshape(features_tmp,
                                     (N, heads, num_out_features))
            # Compute node attention coefficients.
            alpha_src = np.sum(features * att_src, axis=-1)  # shape: N x H
            alpha_dst = np.sum(features * att_dst, axis=-1)  # N x H

            # Calculate attention weights.
            e = np.zeros((num_entries, heads), dtype=dtype)
            softmax_sum = np.zeros((N, heads), dtype=dtype)

            # TODO: Below loop can be flipped.
            for l in dace.map[0:N]:
                for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                    # Calculating e_l->colv
                    colv = columns[v]
                    e_tmp = alpha_src[l] + alpha_dst[colv]
                    # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[v] = e_tmp
                    softmax_sum[colv] += e_tmp

            # Softmax normalization.
            for j in dace.map[0:num_entries]:
                colj = columns[j]
                e[j] = e[j] / softmax_sum[colj]

            # Implementation with loop flattening.
            helper_row = dace.define_local((num_entries,), dtype=dace.int64)
            for l in dace.map[0:N]:
                for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                    helper_row[v] = l

            output[:] = 0
            for i in dace.map[0:num_entries]:
                colv = columns[i]
                b = helper_row[i]
                if heads == 1:
                    output[colv] += e[i] * features[b]
                else:
                    output[colv] += np.reshape(
                        np.reshape(e[i], (heads, 1)) * features[b],
                        (heads * num_out_features,))

        if do_bias:
            def bias_prog(node_features, rowptrs, columns, lin_srcDOTweight,
                          att_src, att_dst, bias, output):
                gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                       att_src, att_dst, output)
                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gat_op


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="csr")
class GATConvCSR(GATConvBase):
    graph_format = sparse.CsrGraph
    input_spec = {
        "node_features": SpecialInputType.VAL_DTYPE,
        "rowptrs": SpecialInputType.IDX_DTYPE,
        "columns": SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        if do_bias:
            def gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                       att_src, att_dst, bias, output):
                """
                node_features: input features, N x F
                rowptrs: rowptr, N+1
                columns: col, num_entries
                lin_srcDOTweight: H * F' x F
                att_src: 1 x H x F'
                att_dst: 1 x H x F'
                output: N x H * F'
                """

                # Compute node attention coefficients.
                # This doesn't work because this einsum is not supported by dace.
                if heads == 1:
                    # Transform input features.
                    # features: N x F'
                    features = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                    alpha_src = features @ att_src[0, 0]
                    alpha_dst = features @ att_dst[0, 0]

                    # Calculate attention weights.
                    e = np.empty((num_entries,), dtype=dtype)
                    softmax_sum = np.zeros((N,), dtype=dtype)

                    for l in dace.map[0:N]:
                        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                            # Calculating e_l->colv
                            colv = columns[v]
                            e[v] = alpha_src[l] + alpha_dst[colv]

                    for j in dace.map[0:num_entries]:
                        colj = columns[j]
                        e[j] = np.exp(np.maximum(negative_slope * e[j], e[j]))
                        softmax_sum[colj] += e[j]

                    # Softmax normalization.
                    for j in dace.map[0:num_entries]:
                        colj = columns[j]
                        e[j] = e[j] / softmax_sum[colj]

                    for i, j in dace.map[0:N, 0:heads * num_out_features]:
                        output[i, j] = bias[j]
                    csrmm(rowptrs, columns, e, features, output,
                          transA=True, beta=1.0)

                else:
                    # Transform input features.
                    features_tmp = np.einsum('ij,kj->ik', node_features,
                                             lin_srcDOTweight)
                    # features: N x H x F'
                    features = np.reshape(features_tmp,
                                          (N, heads, num_out_features))

                    # This ends up ridiculously slow because the outer loop is
                    # executed on gpu and everything inside is executed
                    # sequentially. The loop is moved to Sequential and the
                    # inside matmul to GPU in my_auto_optimize.py.

                    features_perm = np.empty((heads, N, num_out_features), dtype=dtype)
                    for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                        features_perm[j, i, k] = features[i, j, k]

                    alpha_src = dace.define_local((heads, N,), dtype=dtype)
                    alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                    for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                        alpha_src[h] = features_perm[h] @ att_src[0, h]
                        alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                    # Calculate attention weights.
                    e = np.empty((heads, num_entries), dtype=dtype)
                    softmax_sum = np.zeros((heads, N), dtype=dtype)

                    for h, l in dace.map[0:heads, 0:N]:
                        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                            # Calculating e_l->colv
                            colv = columns[v]
                            e[h, v] = alpha_src[h, l] + alpha_dst[h, colv]

                    for h, j in dace.map[0:heads, 0:num_entries]:
                        colj = columns[j]
                        e[h, j] = np.exp(np.maximum(negative_slope * e[h, j], e[h, j]))
                        softmax_sum[h, colj] += e[h, j]

                    # Softmax normalization.
                    for h, j in dace.map[0:heads, 0:num_entries]:
                        colj = columns[j]
                        e[h, j] = e[h, j] / softmax_sum[h, colj]

                    output_perm = np.zeros((heads, N, num_out_features),
                                           dtype=dtype)  # H x N x F'

                    # This results in incorrect code (exceeding the max grid size).
                    # features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'

                    # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                    for h in range(heads):
                        csrmm(rowptrs, columns, e[h], features_perm[h],
                              output_perm[h],
                              transA=True,
                              beta=1.0)

                    # output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
                    #                        (N, heads * num_out_features))

                    for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                        output[i, j * num_out_features + k] = (
                                output_perm[j, i, k]
                                + bias[j * num_out_features + k])
        else:
            def gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                       att_src, att_dst, output):
                """
                node_features: input features, N x F
                rowptrs: rowptr, N+1
                columns: col, num_entries
                lin_srcDOTweight: H * F' x F
                att_src: 1 x H x F'
                att_dst: 1 x H x F'
                output: N x H * F'
                """

                # Compute node attention coefficients.
                if heads == 1:
                    # Transform input features.
                    # features: N x F'
                    features = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                    alpha_src = features @ att_src[0, 0]
                    alpha_dst = features @ att_dst[0, 0]

                    # Calculate attention weights.
                    e = np.empty((num_entries,), dtype=dtype)
                    softmax_sum = np.zeros((N,), dtype=dtype)

                    for l in dace.map[0:N]:
                        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                            # Calculating e_l->colv
                            colv = columns[v]
                            e[v] = alpha_src[l] + alpha_dst[colv]

                    for j in dace.map[0:num_entries]:
                        colj = columns[j]
                        e[j] = np.exp(np.maximum(negative_slope * e[j], e[j]))
                        softmax_sum[colj] += e[j]

                    # Softmax normalization.
                    for j in dace.map[0:num_entries]:
                        colj = columns[j]
                        e[j] = e[j] / softmax_sum[colj]

                    csrmm(rowptrs, columns, e, features, output,
                          transA=True, beta=0.0)

                else:
                    # Transform input features.
                    features_tmp = np.einsum('ij,kj->ik', node_features,
                                             lin_srcDOTweight)
                    # features: N x H x F'
                    features = np.reshape(features_tmp,
                                          (N, heads, num_out_features))

                    # This ends up ridiculously slow because the outer loop is
                    # executed on gpu and everything inside is executed
                    # sequentially. The loop is moved to Sequential and the
                    # inside matmul to GPU in my_auto_optimize.py.

                    features_perm = np.empty((heads, N, num_out_features),
                                             dtype=dtype)
                    for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                        features_perm[j, i, k] = features[i, j, k]

                    alpha_src = dace.define_local((heads, N,), dtype=dtype)
                    alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                    for h in dace.map[
                             0:heads] @ dace.dtypes.ScheduleType.Sequential:
                        alpha_src[h] = features_perm[h] @ att_src[0, h]
                        alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                    # Calculate attention weights.
                    e = np.empty((heads, num_entries), dtype=dtype)
                    softmax_sum = np.zeros((heads, N), dtype=dtype)

                    for h, l in dace.map[0:heads, 0:N]:
                        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                            # Calculating e_l->colv
                            colv = columns[v]
                            e[h, v] = alpha_src[h, l] + alpha_dst[h, colv]

                    for h, j in dace.map[0:heads, 0:num_entries]:
                        colj = columns[j]
                        e[h, j] = np.exp(
                            np.maximum(negative_slope * e[h, j], e[h, j]))
                        softmax_sum[h, colj] += e[h, j]

                    # Softmax normalization.
                    for h, j in dace.map[0:heads, 0:num_entries]:
                        colj = columns[j]
                        e[h, j] = e[h, j] / softmax_sum[h, colj]

                    output_perm = np.zeros((heads, N, num_out_features),
                                           dtype=dtype)  # H x N x F'

                    # This results in incorrect code (exceeding the max grid size).
                    # features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'

                    # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                    for h in range(heads):
                        csrmm(rowptrs, columns, e[h], features_perm[h],
                              output_perm[h],
                              transA=True,
                              beta=1.0)

                    for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                        output[i, j * num_out_features + k] = output_perm[
                            j, i, k]

        return gat_op


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="csr_stable")
class GATConvCSRStable(GATConvBase):
    graph_format = sparse.CsrGraph
    input_spec = {
        "node_features": SpecialInputType.VAL_DTYPE,
        "rowptrs": SpecialInputType.IDX_DTYPE,
        "columns": SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                   att_src, att_dst, output):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: 1 x H x F'
            att_dst: 1 x H x F'
            output: N x H * F'
            """

            # Transform input features.
            features = dace.define_local((N, heads, num_out_features),
                                         dtype=dtype)
            features_tmp = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)

            # features: N x H x F'
            features[:] = np.reshape(features_tmp,
                                     (N, heads, num_out_features))

            alpha_src = np.sum(features * att_src, axis=-1)  # shape: N x H
            alpha_dst = np.sum(features * att_dst, axis=-1)  # N x H

            # Calculate attention weights.
            e = np.zeros((num_entries, heads), dtype=dtype)
            softmax_sum = np.zeros((N, heads), dtype=dtype)
            softmax_max = np.ones((N, heads), dtype=dtype) * -np.inf

            for l in dace.map[0:N]:
                for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                    # Calculating e_l->colv
                    colv = columns[v]
                    e_tmp = alpha_src[l] + alpha_dst[colv]
                    # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[v] = e_tmp
                    softmax_max[colv] = np.maximum(softmax_max[colv], e[v])

            # TODO: sequential map, otherwise incorrect with autoopt.
            for j in dace.map[
                     0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                colj = columns[j]
                e[j] = np.exp(e[j] - softmax_max[colj])
                softmax_sum[colj] += e[j]

            # Softmax normalization.
            for j in dace.map[0:num_entries]:
                colj = columns[j]
                e[j] = e[j] / softmax_sum[colj]

            if heads == 1:
                csrmm(rowptrs, columns, e,
                      np.reshape(features, (N, num_out_features)), output,
                      transA=True, beta=0.0)
            else:
                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'
                features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'
                e_perm = np.transpose(e, (1, 0))  # H x num_entries
                # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                for h in range(heads):
                    csrmm(rowptrs, columns, e_perm[h], features_perm[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
                                       (N, heads * num_out_features))

        if do_bias:
            def bias_prog(node_features, rowptrs, columns, lin_srcDOTweight,
                          att_src, att_dst, bias, output):
                gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                       att_src, att_dst,
                       output)
                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gat_op


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="coo")
class GATConvCOO(GATConvBase):
    graph_format = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, bias, output):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F'
            att_dst: H x F'
            output: N x H * F'
            """

            if heads == 1:
                # Transform input features.
                features = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
                alpha_src = features @ att_src[0, 0]
                alpha_dst = features @ att_dst[0, 0]

                # Calculate attention weights.
                e = np.empty((num_entries,), dtype=dtype)
                softmax_sum = np.zeros((N,), dtype=dtype)

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    # TODO: alpha_src gets read num_entries times, not N!
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[i] = e_tmp

                    # # TODO: This is a workaround. With no schedule type, the results are incorrect
                    #  on CPU with autoopt.
                    # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
                    # col = columns[i]
                    softmax_sum[col] += e[i]

                # Softmax normalization.
                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, e, features, output, transA=True, beta=1.0)

            else:
                # Transform input features.
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                # features: N x H x F'
                features = np.reshape(features_tmp,
                                      (N, heads, num_out_features))

                # This ends up ridiculously slow because the outer loop is
                # executed on gpu and everything inside is executed
                # sequentially. The loop is moved to Sequential and the
                # inside matmul to GPU in my_auto_optimize.py.

                features_perm = np.transpose(features, (1, 0, 2))

                alpha_src = dace.define_local((heads, N,), dtype=dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_perm[h] @ att_src[0, h]
                    alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                # Calculate attention weights.
                e = np.empty((heads, num_entries), dtype=dtype)
                softmax_sum = np.zeros((N, heads), dtype=dtype)

                for h, i in dace.map[0:heads, 0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[h, i] = e_tmp

                    softmax_sum[col, h] += e[h, i]

                # Softmax normalization.
                for h, j in dace.map[0:heads, 0:num_entries]:
                    colj = columns[j]
                    e[h, j] = e[h, j] / softmax_sum[colj, h]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'

                # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                for h in range(heads):
                    coomm(rows, columns, e[h], features_perm[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    output[i, j * num_out_features + k] = (
                            output_perm[j, i, k]
                            + bias[j * num_out_features + k])

        if do_bias:
            return gat_op
        else:
            raise NotImplementedError


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="coo_stable")
class GATConvCOOStable(GATConvBase):
    graph_format = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, bias, output):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F'
            att_dst: H x F'
            output: N x H * F'
            """

            if heads == 1:
                # Transform input features.
                features = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
                alpha_src = features @ att_src[0, 0]
                alpha_dst = features @ att_dst[0, 0]

                # Calculate attention weights.
                e = np.empty((num_entries,), dtype=dtype)
                softmax_sum = np.zeros((N,), dtype=dtype)
                softmax_max = np.ones((N,), dtype=dtype) * -np.inf

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    # TODO: alpha_src gets read num_entries times, not N!
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[i] = e_tmp

                    # # TODO: This is a workaround. With no schedule type, the results are incorrect
                    #  on CPU with autoopt.
                    # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
                    # col = columns[i]
                    softmax_max[col] = max(e[i], softmax_max[col])

                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[j] = np.exp(e[j] - softmax_max[col])
                    softmax_sum[col] += e[j]

                # Softmax normalization.
                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, e, features, output, transA=True, beta=1.0)

            else:
                # Transform input features.
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                # features: N x H x F'
                features = np.reshape(features_tmp,
                                      (N, heads, num_out_features))

                # This ends up ridiculously slow because the outer loop is
                # executed on gpu and everything inside is executed
                # sequentially. The loop is moved to Sequential and the
                # inside matmul to GPU in my_auto_optimize.py.

                features_perm = np.transpose(features, (1, 0, 2))

                alpha_src = dace.define_local((heads, N,), dtype=dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_perm[h] @ att_src[0, h]
                    alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                # Calculate attention weights.
                e = np.empty((heads, num_entries), dtype=dtype)
                softmax_sum = np.zeros((N, heads), dtype=dtype)
                softmax_max = np.ones((N, heads), dtype=dtype) * -np.inf

                for h, i in dace.map[0:heads, 0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[h, i] = e_tmp
                    softmax_max[col, h] = max(e[h, i], softmax_max[col, h])

                for h, j in dace.map[0:heads, 0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[h, j] = np.exp(e[h, j] - softmax_max[col, h])
                    softmax_sum[col, h] += e[h, j]

                # Softmax normalization.
                for h, j in dace.map[0:heads, 0:num_entries]:
                    colj = columns[j]
                    e[h, j] = e[h, j] / softmax_sum[colj, h]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'

                # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                for h in range(heads):
                    coomm(rows, columns, e[h], features_perm[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    output[i, j * num_out_features + k] = (
                            output_perm[j, i, k]
                            + bias[j * num_out_features + k])

        if do_bias:
            return gat_op
        else:
            raise NotImplementedError

@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="coo_cached")
class GATConvCOOCached(GATConvBase):
    graph_format = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    buffer_spec: List[ArraySpec] = [
        ArraySpec(name='e', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  module.heads, inputs[1].shape[0])),
        ArraySpec(name='is_pos_C_vals', dtype=dace.bool,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  module.heads, inputs[1].shape[0])),
        ArraySpec(name='features_saved', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[0].shape[0], module.lin_src.weight.shape[0]) if module.heads == 1 else
                  (module.heads, inputs[0].shape[0], module.lin_src.weight.shape[0])
                  ),

    ]

    @staticmethod
    def ssi_fn(ssi: 'SymbolicShapeInference', node: 'NodeProto') -> None:
        op_attributes = {
            attribute_proto.name: convert_attribute_proto(attribute_proto)
            for attribute_proto in node.attribute
        }
        _, module = ssi.placeholder_id_to_module[op_attributes['module_id']]
        output_dtype = ssi.known_vi_[node.input[0]].type.tensor_type.elem_type

        # Output of the node are: output, e, is_pos_C_vals, features_saved
        N, F_in = ssi._get_shape(node, 0)
        heads = module.heads
        F_out = module.out_channels
        num_entries = ssi._get_shape(node, 1)[0]  # rows.shape[0]

        # Output shape.
        out_shape = (N, heads * F_out)
        vi = ssi.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], output_dtype, out_shape))

        # Attention weight shape is (num_entries,) or (heads, num_entries)
        vi = ssi.known_vi_[node.output[1]]
        e_shape = (num_entries,) if heads == 1 else (heads, num_entries)
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[1], output_dtype, e_shape))

        # C_vals is also (num_entries,) or (heads, num_entries), but boolean.
        vi = ssi.known_vi_[node.output[2]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[2], onnx.TensorProto.BOOL, e_shape))

        # features_saved: N x F_out
        features_shape = (N, F_out) if heads == 1 else (heads, N, F_out)
        vi = ssi.known_vi_[node.output[3]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[3], output_dtype, features_shape))

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, bias, output, e, is_pos_C_vals, features_saved):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F'
            att_dst: H x F'
            output: N x H * F'
            """

            if heads == 1:
                # Transform input features.
                features_saved[:] = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
                alpha_src = features_saved @ att_src[0, 0]
                alpha_dst = features_saved @ att_dst[0, 0]

                # Calculate attention weights.
                softmax_sum = np.zeros((N,), dtype=dtype)

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    # TODO: alpha_src gets read num_entries times, not N!
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    is_pos_C_vals[i] = e_tmp > 0
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[i] = e_tmp

                    softmax_sum[col] += e[i]

                # Softmax normalization.
                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, e, features_saved, output, transA=True, beta=1.0)

            else:
                # Transform input features.
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                # features: N x H x F'
                features = np.reshape(features_tmp,
                                      (N, heads, num_out_features))

                # This ends up ridiculously slow because the outer loop is
                # executed on gpu and everything inside is executed
                # sequentially. The loop is moved to Sequential and the
                # inside matmul to GPU in my_auto_optimize.py.

                features_saved[:] = np.transpose(features, (1, 0, 2)) # heads x N x F_out

                alpha_src = dace.define_local((heads, N,), dtype=dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_saved[h] @ att_src[0, h]
                    alpha_dst[h] = features_saved[h] @ att_dst[0, h]

                # Calculate attention weights.
                softmax_sum = np.zeros((N, heads), dtype=dtype)

                for h, i in dace.map[0:heads, 0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    is_pos_C_vals[h, i] = e_tmp > 0
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[h, i] = e_tmp
                    softmax_sum[col, h] += e[h, i]

                # Softmax normalization.
                for h, j in dace.map[0:heads, 0:num_entries]:
                    colj = columns[j]
                    e[h, j] = e[h, j] / softmax_sum[colj, h]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'

                # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                for h in range(heads):
                    coomm(rows, columns, e[h], features_saved[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    output[i, j * num_out_features + k] = (
                            output_perm[j, i, k]
                            + bias[j * num_out_features + k])

        if do_bias:
            return gat_op
        else:
            raise NotImplementedError

@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="coo_stable_cached")
class GATConvCOOStableCached(GATConvBase):
    graph_format = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    buffer_spec: List[ArraySpec] = [
        ArraySpec(name='e', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  module.heads, inputs[1].shape[0])),
        ArraySpec(name='is_pos_C_vals', dtype=dace.bool,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  module.heads, inputs[1].shape[0])),
        ArraySpec(name='features_saved', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[0].shape[0], module.lin_src.weight.shape[0]) if module.heads == 1 else
                  (module.heads, inputs[0].shape[0], module.lin_src.weight.shape[0])
                  ),

    ]

    @staticmethod
    def ssi_fn(ssi: 'SymbolicShapeInference', node: 'NodeProto') -> None:
        op_attributes = {
            attribute_proto.name: convert_attribute_proto(attribute_proto)
            for attribute_proto in node.attribute
        }
        _, module = ssi.placeholder_id_to_module[op_attributes['module_id']]
        output_dtype = ssi.known_vi_[node.input[0]].type.tensor_type.elem_type

        # Output of the node are: output, e, is_pos_C_vals, features_saved
        N, F_in = ssi._get_shape(node, 0)
        heads = module.heads
        F_out = module.out_channels
        num_entries = ssi._get_shape(node, 1)[0]  # rows.shape[0]

        # Output shape.
        out_shape = (N, heads * F_out)
        vi = ssi.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], output_dtype, out_shape))

        # Attention weight shape is (num_entries,) or (heads, num_entries)
        vi = ssi.known_vi_[node.output[1]]
        e_shape = (num_entries,) if heads == 1 else (heads, num_entries)
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[1], output_dtype, e_shape))

        # C_vals is also (num_entries,) or (heads, num_entries), but boolean.
        vi = ssi.known_vi_[node.output[2]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[2], onnx.TensorProto.BOOL, e_shape))

        # features_saved: N x F_out
        features_shape = (N, F_out) if heads == 1 else (heads, N, F_out)
        vi = ssi.known_vi_[node.output[3]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[3], output_dtype, features_shape))

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, bias, output, e, is_pos_C_vals, features_saved):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F'
            att_dst: H x F'
            output: N x H * F'
            """

            if heads == 1:
                # Transform input features.
                features_saved[:] = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
                alpha_src = features_saved @ att_src[0, 0]
                alpha_dst = features_saved @ att_dst[0, 0]

                # Calculate attention weights.
                softmax_sum = np.zeros((N,), dtype=dtype)
                softmax_max = np.ones((N,), dtype=dtype) * -np.inf

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    is_pos_C_vals[i] = e_tmp > 0
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[i] = e_tmp
                    softmax_max[col] = max(e[i], softmax_max[col])

                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[j] = np.exp(e[j] - softmax_max[col])
                    softmax_sum[col] += e[j]

                # Softmax normalization.
                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, e, features_saved, output, transA=True, beta=1.0)

            else:
                # Transform input features.
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                # features: N x H x F'
                features = np.reshape(features_tmp,
                                      (N, heads, num_out_features))

                # This ends up ridiculously slow because the outer loop is
                # executed on gpu and everything inside is executed
                # sequentially. The loop is moved to Sequential and the
                # inside matmul to GPU in my_auto_optimize.py.

                features_saved[:] = np.transpose(features, (1, 0, 2)) # heads x N x F_out

                alpha_src = dace.define_local((heads, N,), dtype=dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_saved[h] @ att_src[0, h]
                    alpha_dst[h] = features_saved[h] @ att_dst[0, h]

                # Calculate attention weights.
                softmax_sum = np.zeros((N, heads), dtype=dtype)
                softmax_max = np.ones((N, heads), dtype=dtype) * -np.inf

                for h, i in dace.map[0:heads, 0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[h, i] = e_tmp
                    softmax_max[col, h] = max(e[h, i], softmax_max[col, h])

                for h, j in dace.map[0:heads, 0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[h, j] = np.exp(e[h, j] - softmax_max[col, h])
                    softmax_sum[col, h] += e[h, j]

                # Softmax normalization.
                for h, j in dace.map[0:heads, 0:num_entries]:
                    colj = columns[j]
                    e[h, j] = e[h, j] / softmax_sum[colj, h]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'

                for h in range(heads):
                    coomm(rows, columns, e[h], features_saved[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    output[i, j * num_out_features + k] = (
                            output_perm[j, i, k]
                            + bias[j * num_out_features + k])

        if do_bias:
            return gat_op
        else:
            raise NotImplementedError


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="coo_stable_cached_altspmm")
class GATConvCOOStableCachedAltSpmm(GATConvBase):
    graph_format = sparse.CooGraph
    input_spec: Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    buffer_spec: List[ArraySpec] = [
        ArraySpec(name='e', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  inputs[1].shape[0], module.heads)),
        ArraySpec(name='is_pos_C_vals', dtype=dace.bool,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[1].shape[0],) if module.heads == 1 else (
                  inputs[1].shape[0], module.heads)),
        ArraySpec(name='features_saved', dtype=dace.float32,
                  # Attention weights has the same shape as `rows`.
                  torch_shape_fn_from_module=lambda module: lambda *inputs: (
                      inputs[0].shape[0], module.lin_src.weight.shape[0]) if module.heads == 1 else
                  (module.heads, inputs[0].shape[0], module.lin_src.weight.shape[0])
                  ),

    ]

    @staticmethod
    def ssi_fn(ssi: 'SymbolicShapeInference', node: 'NodeProto') -> None:
        op_attributes = {
            attribute_proto.name: convert_attribute_proto(attribute_proto)
            for attribute_proto in node.attribute
        }
        _, module = ssi.placeholder_id_to_module[op_attributes['module_id']]
        output_dtype = ssi.known_vi_[node.input[0]].type.tensor_type.elem_type

        # Output of the node are: output, e, is_pos_C_vals, features_saved
        N, F_in = ssi._get_shape(node, 0)
        heads = module.heads
        F_out = module.out_channels
        num_entries = ssi._get_shape(node, 1)[0]  # rows.shape[0]

        # Output shape.
        out_shape = (N, heads * F_out)
        vi = ssi.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], output_dtype, out_shape))

        # Attention weight shape is (num_entries,) or (heads, num_entries)
        vi = ssi.known_vi_[node.output[1]]
        e_shape = (num_entries,) if heads == 1 else (num_entries, heads)
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[1], output_dtype, e_shape))

        # C_vals is also (num_entries,) or (heads, num_entries), but boolean.
        vi = ssi.known_vi_[node.output[2]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[2], onnx.TensorProto.BOOL, e_shape))

        # features_saved: N x F_out
        features_shape = (N, F_out) if heads == 1 else (heads, N, F_out)
        vi = ssi.known_vi_[node.output[3]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[3], output_dtype, features_shape))

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, bias, output, e, is_pos_C_vals, features_saved):
            """
            node_features: input features, N x F
            rowptrs: rowptr, N+1
            columns: col, num_entries
            lin_srcDOTweight: H * F' x F
            att_src: H x F'
            att_dst: H x F'
            output: N x H * F'
            """

            if heads == 1:
                # Transform input features.
                features_saved[:] = np.einsum('ij,kj->ik', node_features,
                                     lin_srcDOTweight)
                alpha_src = features_saved @ att_src[0, 0]
                alpha_dst = features_saved @ att_dst[0, 0]

                # Calculate attention weights.
                softmax_sum = np.zeros((N,), dtype=dtype)
                softmax_max = np.ones((N,), dtype=dtype) * -np.inf

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    is_pos_C_vals[i] = e_tmp > 0
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[i] = e_tmp
                    softmax_max[col] = max(e[i], softmax_max[col])

                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[j] = np.exp(e[j] - softmax_max[col])
                    softmax_sum[col] += e[j]

                # Softmax normalization.
                for j in dace.map[0:num_entries] @ dace.dtypes.ScheduleType.Sequential:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] = bias[j]
                coomm(rows, columns, e, features_saved, output, transA=True, beta=1.0)

            else:
                # Transform input features.
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)
                # features: N x H x F'
                features = np.reshape(features_tmp,
                                      (N, heads, num_out_features))

                # This ends up ridiculously slow because the outer loop is
                # executed on gpu and everything inside is executed
                # sequentially. The loop is moved to Sequential and the
                # inside matmul to GPU in my_auto_optimize.py.

                features_saved[:] = np.transpose(features, (1, 0, 2)) # heads x N x F_out

                alpha_src = dace.define_local((heads, N,), dtype=dtype)
                alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    alpha_src[h] = features_saved[h] @ att_src[0, h]
                    alpha_dst[h] = features_saved[h] @ att_dst[0, h]

                # Calculate attention weights.
                softmax_sum = np.zeros((N, heads), dtype=dtype)
                softmax_max = np.ones((N, heads), dtype=dtype) * -np.inf

                for i, h in dace.map[0:num_entries, 0:heads]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[h, row] + alpha_dst[h, col]
                    # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e[i, h] = e_tmp
                    softmax_max[col, h] = max(e[i, h], softmax_max[col, h])

                for j, h in dace.map[0:num_entries, 0:heads] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[j]
                    e[j, h] = np.exp(e[j, h] - softmax_max[col, h])
                    softmax_sum[col, h] += e[j, h]

                # Softmax normalization.
                for j, h in dace.map[0:num_entries, 0:heads]:
                    colj = columns[j]
                    e[j, h] = e[j, h] / softmax_sum[colj, h]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'


                for i, h, k in dace.map[0:num_entries, 0:heads, 0:num_out_features] @ dace.dtypes.ScheduleType.Sequential:
                    col = columns[i]
                    row = rows[i]
                    mult = e[i, h] * features_saved[h, row, k]
                    output_perm[h, col, k] += mult

                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    output[i, j * num_out_features + k] = (
                            output_perm[j, i, k]
                            + bias[j * num_out_features + k])

        if do_bias:
            return gat_op
        else:
            raise NotImplementedError

@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="csc")
class GATConvCSC(GATConvBase):
    graph_format = sparse.CscGraph
    input_spec = {
        "node_features": SpecialInputType.VAL_DTYPE,
        "colptrs": SpecialInputType.IDX_DTYPE,
        "rows": SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        if do_bias:
            def gat_op(node_features, colptrs, rows, lin_srcDOTweight,
                       att_src, att_dst, bias, output):
                """
                node_features: input features, N x F
                rowptrs: rowptr, N+1
                columns: col, num_entries
                lin_srcDOTweight: H * F' x F
                att_src: 1 x H x F'
                att_dst: 1 x H x F'
                output: N x H * F'
                """

                # Compute node attention coefficients.
                # This doesn't work because this einsum is not supported by dace.
                if heads == 1:
                    # Transform input features.
                    # features: N x F'
                    features = dace.define_local((N, num_out_features),
                                                 dtype=dtype)
                    features[:] = np.einsum('ij,kj->ik', node_features,
                                            lin_srcDOTweight)
                    alpha_src = features @ att_src[0, 0]
                    alpha_dst = features @ att_dst[0, 0]

                    # Calculate attention weights.
                    e = np.empty((num_entries,), dtype=dtype)
                    softmax_sum = np.zeros((N,), dtype=dtype)

                    for l in dace.map[0:N]:
                        for v in dace.map[colptrs[l]:colptrs[l + 1]]:
                            # Calculating e_l->colv
                            row = rows[v]
                            e[v] = alpha_src[row] + alpha_dst[l]
                            e[v] = np.exp(np.maximum(negative_slope * e[v], e[v]))
                            softmax_sum[l] += e[v]

                    # # Softmax normalization.
                    for l in dace.map[0:N] @ dace.dtypes.ScheduleType.Sequential:
                        for v in dace.map[colptrs[l]:colptrs[l + 1]]:
                            e[v] = e[v] / softmax_sum[l]

                    for i, j in dace.map[0:N, 0:heads * num_out_features]:
                        output[i, j] = bias[j]

                    # A is in CSC format, so in order to compute A.t @ B, we call CSR
                    # matmul. This is because CSC(A) = CSR(A.t).
                    csrmm(colptrs, rows, e, features, output,
                          transA=False, beta=1.0)

                else:
                    # Transform input features.
                    features_tmp = np.einsum('ij,kj->ik', node_features,
                                             lin_srcDOTweight)
                    # features: N x H x F'
                    features = np.reshape(features_tmp,
                                          (N, heads, num_out_features))

                    # This ends up ridiculously slow because the outer loop is
                    # executed on gpu and everything inside is executed
                    # sequentially. The loop is moved to Sequential and the
                    # inside matmul to GPU in my_auto_optimize.py.

                    features_perm = np.transpose(features, (1, 0, 2))

                    alpha_src = dace.define_local((heads, N,), dtype=dtype)
                    alpha_dst = dace.define_local((heads, N,), dtype=dtype)
                    for h in dace.map[0:heads] @ dace.dtypes.ScheduleType.Sequential:
                        alpha_src[h] = features_perm[h] @ att_src[0, h]
                        alpha_dst[h] = features_perm[h] @ att_dst[0, h]

                    # Calculate attention weights.
                    e = np.empty((heads, num_entries), dtype=dtype)
                    softmax_sum = np.zeros((heads, N), dtype=dtype)

                    for h, l in dace.map[0:heads, 0:N]:
                        for v in dace.map[colptrs[l]:colptrs[l + 1]]:
                            # Calculating e_l->colv
                            row = rows[v]
                            e[h, v] = alpha_src[h, row] + alpha_dst[h, l]
                            e[h, v] = np.exp(np.maximum(negative_slope * e[h, v], e[h, v]))
                            softmax_sum[h, l] += e[h, v]

                    # Softmax normalization.
                    # Schedule is needed because otherwise we get wrong results on CPU ;(
                    # The schedule is ignored when applying GPU transformations,
                    # so it's fine.
                    for h, l in dace.map[0:heads, 0:N] @ dace.dtypes.ScheduleType.Sequential:
                        for v in dace.map[colptrs[l]:colptrs[l + 1]]:
                            e[h, v] = e[h, v] / softmax_sum[h, l]

                    output_perm = np.zeros((heads, N, num_out_features),
                                           dtype=dtype)  # H x N x F'

                    # This results in incorrect code (exceeding the max grid size).
                    # features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'

                    # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                    for h in range(heads):
                        csrmm(colptrs, rows, e[h], features_perm[h],
                              output_perm[h],
                              transA=False,
                              beta=1.0)

                    # output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
                    #                        (N, heads * num_out_features))

                    for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                        output[i, j * num_out_features + k] = (
                                output_perm[j, i, k]
                                + bias[j * num_out_features + k])

            return gat_op
        else:
            raise NotImplementedError
