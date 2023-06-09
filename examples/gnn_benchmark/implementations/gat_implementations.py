import abc
import typing

import dace
import numpy as np
from dace import nodes, SDFG, SDFGState

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


class GATConvBase(SparseLayerBase, metaclass=abc.ABCMeta):
    @classmethod
    def forward(cls, node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        if node.module.add_self_loops:
            raise NotImplementedError("Adding self loops is not supported. "
                                      "Add self-loops in preprocessing.")

        features_desc = in_desc_with_name(node, state, sdfg, "node_features")
        N, num_in_features = features_desc.shape
        dtype = features_desc.dtype

        col_desc = in_desc_with_name(node, state, sdfg, "columns")
        num_entries, = col_desc.shape

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
    input_spec: typing.Dict[str, dace.dtypes.typeclass] = {
        'node_features': common.SpecialInputType.VAL_DTYPE,
        'rows': common.SpecialInputType.IDX_DTYPE,
        'columns': common.SpecialInputType.IDX_DTYPE,
    }

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        @dace.program
        def gat_op(node_features, rows, columns, lin_srcDOTweight,
                   att_src, att_dst, output):
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
                features = dace.define_local((N, heads, num_out_features),
                                             dtype=dtype)
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)

                # features: N x H x F'
                features[:] = np.reshape(features_tmp,
                                         (N, heads, num_out_features))

                alpha_src_tmp = dace.define_local((N, heads, num_out_features),
                                                  dtype=dtype)
                alpha_dst_tmp = dace.define_local((N, heads, num_out_features),
                                                  dtype=dtype)
                # Compute node attention coefficients.
                # features * att_src: N x H x F
                for k, j, i in dace.map[0:num_out_features, 0:heads, 0:N]:
                    alpha_src_tmp[i, j, k] = features[i, j, k] * att_src[0, j, k]
                    alpha_dst_tmp[i, j, k] = features[i, j, k] * att_dst[0, j, k]

                alpha_src = np.sum(alpha_src_tmp, axis=-1)  # shape: N x H
                alpha_dst = np.sum(alpha_dst_tmp, axis=-1)  # N x H

                # Calculate attention weights.
                e = np.empty((num_entries, heads), dtype=dtype)
                softmax_sum = np.zeros((N, heads), dtype=dtype)

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[i] = e_tmp

                    # # TODO: This is a workaround. With no schedule type, the results are incorrect with autoopt
                    # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
                    # col = columns[i]
                    softmax_sum[col] += e[i]

                # Softmax normalization.
                for j in dace.map[0:num_entries]:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                coomm(rows, columns, e,
                      features[:, 0, :], output,
                      transA=True, beta=0.0)

            else:
                # Transform input features.
                features = dace.define_local((N, heads, num_out_features),
                                             dtype=dtype)
                features_tmp = np.einsum('ij,kj->ik', node_features,
                                         lin_srcDOTweight)

                # features: N x H x F'
                features[:] = np.reshape(features_tmp,
                                         (N, heads, num_out_features))

                alpha_src_tmp = dace.define_local((N, heads, num_out_features),
                                                  dtype=dtype)
                alpha_dst_tmp = dace.define_local((N, heads, num_out_features),
                                                  dtype=dtype)
                # Compute node attention coefficients.
                # features * att_src: N x H x F
                for k, j, i in dace.map[0:num_out_features, 0:heads, 0:N]:
                    alpha_src_tmp[i, j, k] = features[i, j, k] * att_src[0, j, k]
                    alpha_dst_tmp[i, j, k] = features[i, j, k] * att_dst[0, j, k]

                alpha_src = np.sum(alpha_src_tmp, axis=-1)  # shape: N x H
                alpha_dst = np.sum(alpha_dst_tmp, axis=-1)  # N x H

                # Calculate attention weights.
                e = np.empty((num_entries, heads), dtype=dtype)
                softmax_sum = np.zeros((N, heads), dtype=dtype)

                for i in dace.map[0:num_entries]:
                    row = rows[i]
                    col = columns[i]
                    e_tmp = alpha_src[row] + alpha_dst[col]
                    # # LeakyReLU
                    e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                    e_tmp = np.exp(e_tmp)
                    e[i] = e_tmp

                    # # TODO: This is a workaround. With no schedule type, the results are incorrect with autoopt
                    # for i in dace.map[0:num_entries]@dace.dtypes.ScheduleType.Sequential:
                    # col = columns[i]
                    softmax_sum[col] += e[i]

                # Softmax normalization.
                for j in dace.map[0:num_entries]:
                    colj = columns[j]
                    e[j] = e[j] / softmax_sum[colj]

                output_perm = np.zeros((heads, N, num_out_features),
                                       dtype=dtype)  # H x N x F'
                # features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'
                features_perm = dace.define_local((heads, N, num_out_features),
                                                  dtype=dtype)
                for j, i, k in dace.map[0:heads, 0:N, 0:num_out_features]:
                    features_perm[j, i, k] = features[i, j, k]

                e_perm = np.transpose(e, (1, 0))  # H x num_entries
                # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
                for h in range(heads):
                    coomm(rows, columns, e_perm[h], features_perm[h],
                          output_perm[h],
                          transA=True,
                          beta=1.0)

                output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
                                       (N, heads * num_out_features))

        if do_bias:
            def bias_prog(node_features, rows, columns, lin_srcDOTweight,
                          att_src, att_dst, bias, output):
                gat_op(node_features, rows, columns, lin_srcDOTweight,
                       att_src, att_dst, output)
                for i, j in dace.map[0:N, 0:heads * num_out_features]:
                    output[i, j] += bias[j]

            return bias_prog
        return gat_op
