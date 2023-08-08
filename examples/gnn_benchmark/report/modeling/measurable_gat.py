from typing import List

import dace

from examples.gnn_benchmark.report.modeling.measurable_layers import CompositeOp
from examples.gnn_benchmark.report.modeling.measurable_ops import MeasurableOp, Matmul, Csrmm, \
    BatchedMatmul, AddBias, Permute, PermuteAndAddBias, Coomm, MultiheadCooSddmm, Reduce


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


class CooSoftmaxBackwardDotProds(MeasurableOp):
    def __init__(self, heads: int, num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.heads = heads
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype

    def flops(self):
        return self.heads * self.num_entries

    def min_memory(self):
        val_count = self.heads * self.num_entries * 2
        idx_count = self.num_entries
        return val_count * self.val_dtype.bytes + idx_count * self.idx_dtype.bytes


class CooSoftmaxAndLeakyReluBackward(MeasurableOp):
    # dE_val[:] = d_alpha_vals[h, i] - dot_prods[col, h] * e[h, i]
    # dC_val = dace.define_local_scalar(val_dtype)
    # dC_val[:] = dE_val * (neg_slope + one_min_neg_slope * is_pos_C_vals[h, i])
    # dr[h, row] += dC_val
    # dl[h, col] += dC_val
    def __init__(self, num_nodes: int, heads: int, num_entries: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.num_nodes = num_nodes
        self.heads = heads
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype

    def flops(self):
        return self.heads * 7 * self.num_entries

    def min_memory(self):
        # Load:
        #   - d_alpha_vals, e (2 * num_entries * heads)
        #   - dot_prods (num_nodes * heads)
        #   - is_pos_C_vals (num_entries * heads)
        #   - rows (num_entries)
        #   - columns (num_entries)
        val_count = self.heads * (2 * self.num_entries + self.num_nodes)
        bool_count = self.heads * self.num_entries
        idx_count = 2 * self.num_entries
        return val_count * self.val_dtype.bytes + idx_count * self.idx_dtype.bytes + bool_count * 1


class CooComputeDHprime(MeasurableOp):
    # for h, i, k in dace.map[0:heads, 0:N,
    #                0:F_out] @ dace.dtypes.ScheduleType.Sequential:
    #     dH_prime[i, h, k] = dH_prime_perm[h, i, k] + dl[h, i] * att_dst[0, h, k] + dr[
    #         h, i] * att_src[0, h, k]
    def __init__(self, num_nodes: int, heads: int, F_out: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.num_nodes = num_nodes
        self.heads = heads
        self.F_out = F_out
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype

    def flops(self):
        return self.heads * 4 * self.num_nodes * self.F_out

    def min_memory(self):
        # Load:
        #   - dH_prime_perm, dH_prime, (2 * num_nodes * heads * F_out)
        #   - att_src, att_dst (2 * heads * F_out)
        #   - dl, dr (2 * num_nodes * heads)
        val_count = self.heads * (
                2 * self.num_nodes * self.F_out + 2 * self.num_nodes + 2 * self.F_out)
        return val_count * self.val_dtype.bytes


class GATConvCOOBackwardNoCache(CompositeOp):
    def __init__(self, num_nodes: int, heads: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool,
                 compute_input_grad: bool):
        self.num_nodes = num_nodes
        self.heads = heads
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias
        self.compute_input_grad = compute_input_grad

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

        compute_gradients = [
            # Compute gradients for d_alpha_vals: SDDMM
            MultiheadCooSddmm(N=num_nodes, M=F_out, heads=heads, nnz=num_entries,
                              val_dtype=val_dtype, idx_dtype=idx_dtype),
            # Compute dot prods for softmax
            CooSoftmaxBackwardDotProds(heads=heads, num_entries=num_entries,
                                       val_dtype=val_dtype, idx_dtype=idx_dtype),
            # Compute softmax backward and leakyrelu backward.
            CooSoftmaxAndLeakyReluBackward(num_nodes=num_nodes, heads=heads,
                                           num_entries=num_entries,
                                           val_dtype=val_dtype,
                                           idx_dtype=idx_dtype),
            # Permute
            Permute(shape=(num_nodes, heads, F_out), val_dtype=val_dtype),

        ]
        one_spmm = Coomm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                         val_dtype=val_dtype, idx_dtype=idx_dtype)

        compute_remaining_grads = [
            # Compute dH_prime
            CooComputeDHprime(num_nodes=num_nodes, heads=heads, F_out=F_out,
                              val_dtype=val_dtype, idx_dtype=idx_dtype),
            # Weight grad
            Matmul(N=heads * F_out, M=num_nodes, F=F_in, val_dtype=val_dtype),
            # bias grad
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype),
            # Attention grads.
            BatchedMatmul(B=heads, N=1, M=num_nodes, F=F_out,
                          val_dtype=val_dtype),
            BatchedMatmul(B=heads, N=1, M=num_nodes, F=F_out,
                          val_dtype=val_dtype),
        ]
        if self.compute_input_grad:
            compute_remaining_grads += [
                # Input grad
                Matmul(N=num_nodes, M=heads * F_out, F=F_in, val_dtype=val_dtype),
            ]

        self.subops = (compute_attention_weights + compute_gradients + self.heads * [one_spmm] +
                       compute_remaining_grads)
