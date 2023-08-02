from typing import List

import dace

from examples.gnn_benchmark.report.modeling.measurable_layers import CompositeOp
from examples.gnn_benchmark.report.modeling.measurable_ops import MeasurableOp, Matmul, Csrmm, \
    BatchedMatmul, AddBias, Permute, PermuteAndAddBias, Coomm


class GATConvCSRAttentionWeights(MeasurableOp):
    def __init__(self, num_nodes: int, heads: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.num_nodes = num_nodes
        self.heads = heads
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype

    def flops(self):
        # For each edge and head:
        #   - sum up alpha_src, alpha_dst
        #   - compute leaky relu (2 ops, mult and max)
        #   - compute softmax (3 ops, add edge to softmax sum, exp and div)
        # Toal 1 + 2 + 3 = 6 ops per edge
        return self.heads * 6 * self.num_entries

    def min_memory(self):
        # Load:
        #   - alpha_src, alpha_dst (2 * num_nodes * heads)
        #   - softmax sum (num_nodes * heads)
        #   - rowptrs (num_nodes + 1)
        #   - columns (num_entries)
        val_count = self.heads * (2 * self.num_nodes + self.num_entries)
        idx_count = self.num_nodes + 1 + self.num_entries
        return val_count * self.val_dtype.bytes + idx_count * self.idx_dtype.bytes


class GATConvCSR(CompositeOp):
    impl_name = 'csr'

    def __init__(self, num_nodes: int, heads: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool):
        self.num_nodes = num_nodes
        self.heads = heads
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        if heads == 1:
            self.subops = [
                # Compute H'
                Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
                # Compute alpha_src, alpha_dst.
                Matmul(N=num_nodes, M=F_out, F=1, val_dtype=val_dtype),
                Matmul(N=num_nodes, M=F_out, F=1, val_dtype=val_dtype),

                # Compute atttention weights.
                GATConvCSRAttentionWeights(num_nodes=num_nodes, heads=heads,
                                           num_entries=num_entries, val_dtype=val_dtype,
                                           idx_dtype=idx_dtype),
                # Compute output.
                Csrmm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                      val_dtype=val_dtype, idx_dtype=idx_dtype),
                AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
            ]
        else:
            self.subops: List[MeasurableOp] = [
                                                  # Compute H'
                                                  Matmul(N=num_nodes, M=F_in, F=heads * F_out,
                                                         val_dtype=val_dtype),
                                                  # Permute features.
                                                  Permute(shape=(num_nodes, heads, F_out),
                                                          val_dtype=val_dtype),
                                                  # Compute alpha_src, alpha_dst.
                                                  BatchedMatmul(B=heads, N=num_nodes, M=F_out, F=1,
                                                                val_dtype=val_dtype),
                                                  BatchedMatmul(B=heads, N=num_nodes, M=F_out, F=1,
                                                                val_dtype=val_dtype),

                                                  # Compute atttention weights.
                                                  GATConvCSRAttentionWeights(num_nodes=num_nodes,
                                                                             heads=heads,
                                                                             num_entries=num_entries,
                                                                             val_dtype=val_dtype,
                                                                             idx_dtype=idx_dtype),
                                              ] + [
                                                  # Compute output.
                                                  Csrmm(N=num_nodes, M=num_nodes, F=F_out,
                                                        nnz=num_entries,
                                                        val_dtype=val_dtype, idx_dtype=idx_dtype),
                                              ] * self.heads + [
                                                  PermuteAndAddBias(shape=(num_nodes, heads, F_out),
                                                                    axis=-1, val_dtype=val_dtype)
                                              ]


class GATConvCOOAttentionWeights(MeasurableOp):
    def __init__(self, num_nodes: int, heads: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.num_nodes = num_nodes
        self.heads = heads
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype

    def flops(self):
        # For each edge and head:
        #   - sum up alpha_src, alpha_dst
        #   - compute leaky relu (2 ops, mult and max)
        #   - compute softmax (3 ops, add edge to softmax sum, exp and div)
        # Toal 1 + 2 + 3 = 6 ops per edge
        return self.heads * 6 * self.num_entries

    def min_memory(self):
        # Load:
        #   - alpha_src, alpha_dst (2 * num_nodes * heads)
        #   - softmax sum (num_nodes * heads)
        #   - rows (num_entries)
        #   - columns (num_entries)
        val_count = self.heads * (2 * self.num_nodes + self.num_entries)
        idx_count = 2 * self.num_entries
        return val_count * self.val_dtype.bytes + idx_count * self.idx_dtype.bytes


class GATConvCOO(CompositeOp):
    impl_name = 'coo'

    def __init__(self, num_nodes: int, heads: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool):
        self.num_nodes = num_nodes
        self.heads = heads
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        if heads == 1:
            self.subops = [
                # Compute H'
                Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
                # Compute alpha_src, alpha_dst.
                Matmul(N=num_nodes, M=F_out, F=1, val_dtype=val_dtype),
                Matmul(N=num_nodes, M=F_out, F=1, val_dtype=val_dtype),

                # Compute atttention weights.
                GATConvCOOAttentionWeights(num_nodes=num_nodes, heads=heads,
                                           num_entries=num_entries, val_dtype=val_dtype,
                                           idx_dtype=idx_dtype),
                # Compute output.
                Coomm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                      val_dtype=val_dtype, idx_dtype=idx_dtype),
                AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
            ]
        else:
            compute_attention_weights: List[MeasurableOp] = [
                # Compute H'
                Matmul(N=num_nodes, M=F_in, F=heads * F_out,
                       val_dtype=val_dtype),
                # Permute features.
                Permute(shape=(num_nodes, heads, F_out),
                        val_dtype=val_dtype),
                # Compute alpha_src, alpha_dst.
                BatchedMatmul(B=heads, N=num_nodes, M=F_out, F=1,
                              val_dtype=val_dtype),
                BatchedMatmul(B=heads, N=num_nodes, M=F_out, F=1,
                              val_dtype=val_dtype),

                # Compute atttention weights.
                GATConvCOOAttentionWeights(num_nodes=num_nodes,
                                           heads=heads,
                                           num_entries=num_entries,
                                           val_dtype=val_dtype,
                                           idx_dtype=idx_dtype),
            ]
            one_spmm: MeasurableOp = Coomm(N=num_nodes, M=num_nodes, F=F_out,
                                           nnz=num_entries,
                                           val_dtype=val_dtype, idx_dtype=idx_dtype)
            compute_output: List[MeasurableOp] = [one_spmm] * self.heads + [
                PermuteAndAddBias(shape=(num_nodes, heads, F_out),
                                  axis=-1, val_dtype=val_dtype),
            ]
            self.subops = compute_attention_weights + compute_output


