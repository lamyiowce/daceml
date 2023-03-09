import typing

import dace
import numpy as np
from dace import nodes, SDFG, SDFGState

from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation
from daceml.onnx.op_implementations.utils import program_for_node
from daceml.util.utils import in_desc_with_name


class GATConvBase(ONNXForward):
    @staticmethod
    def get_input_spec():
        raise NotImplementedError

    @staticmethod
    def make_op(N: int, heads: int, num_out_features: int, num_entries: int,
                dtype: dace.dtypes.Typeclasses, negative_slope: float,
                do_bias: bool):
        raise NotImplementedError

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
    @staticmethod
    def get_input_spec():
        return {
            'node_features': dace.float32,
            'rowptrs': dace.int64,
            'columns': dace.int64
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
                for i, j in dace.map[0:N, 0:num_out_features * heads]:
                    output[i, j] += bias[j]

            return bias_prog
        return gat_op


@op_implementation(op="torch_geometric.nn.conv.gat_conv.GATConv",
                   name="csr")
class GATConvCSR(GATConvBase):
    @staticmethod
    def get_input_spec():
        return {
            'node_features': dace.float32,
            'rowptrs': dace.int64,
            'columns': dace.int64
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

            output[:] = 0
            for l in dace.map[0:N]:
                for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                    colv = columns[v]
                    if heads == 1:
                        output[colv] += e[v] * features[l]
                    else:
                        output[colv] += np.reshape(
                            np.reshape(e[v], (heads, 1)) * features[l],
                            (heads * num_out_features,))

        if do_bias:
            def bias_prog(node_features, rowptrs, columns, lin_srcDOTweight,
                          att_src, att_dst, bias, output):
                gat_op(node_features, rowptrs, columns, lin_srcDOTweight,
                       att_src, att_dst, output)
                for i, j in dace.map[0:N, 0:num_out_features * heads]:
                    output[i, j] += bias[j]

            return bias_prog
        return gat_op
