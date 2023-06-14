import dace

from examples.gnn_benchmark.report.measurable_ops import MeasurableOp, Csrmm, \
    Matmul, Reduce


class GCNConvCSR(MeasurableOp):
    """Estimations for GCNConvCSR node."""
    impl_name = 'csr'

    def __init__(self, num_nodes: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool):
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        self.sub_ops = [
            Csrmm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype)
        ]

    def flops(self):
        if self.do_bias:
            bias_flops = self.F_out * self.num_nodes
        else:
            bias_flops = 0
        return sum([op.flops() for op in self.sub_ops]) + bias_flops

    def min_memory(self):
        if self.do_bias:
            # Read: output (N x F_out), bias (F_out)
            bias_bytes = (
                                 self.num_nodes * self.F_out + self.F_out) * self.val_dtype.bytes
        else:
            bias_bytes = 0
        return sum([op.min_memory() for op in self.sub_ops]) + bias_bytes


class GCNConvCSRAdapt(MeasurableOp):
    """Estimations for GCNConvCSRAdapt node."""
    impl_name = 'csr_adapt'

    def __init__(self, num_nodes: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool):
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        # A.t @ (X @ W) or (A.t @ X) @ W, whichever is cheaper.
        self.sub_ops = [
            Csrmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype)
        ]

    def flops(self):
        if self.do_bias:
            bias_flops = self.F_out * self.num_nodes
        else:
            bias_flops = 0
        return sum([op.flops() for op in self.sub_ops]) + bias_flops

    def min_memory(self):
        if self.do_bias:
            # Read: output (N x F_out), bias (F_out)
            bias_count = self.num_nodes * self.F_out + self.F_out
            bias_bytes = bias_count * self.val_dtype.bytes
        else:
            bias_bytes = 0
        return sum([op.min_memory() for op in self.sub_ops]) + bias_bytes


class BackwardGCNConvCSR(MeasurableOp):
    impl_name = 'csr'

    def __init__(self, num_nodes: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool,
                 compute_input_grad: bool):
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        # Grad W = Grad C^T @ A^t @ X
        weight_grad_subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = A @ Grad G @ W
        input_grad_subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                    val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops

    def flops(self):
        return sum([op.flops() for op in self.subops])

    def min_memory(self):
        return sum([op.min_memory() for op in self.subops])


class BackwardGCNConvCSRAdapt(MeasurableOp):
    impl_name = 'csr'

    def __init__(self, num_nodes: int, F_in: int, F_out: int,
                 num_entries: int, val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass, do_bias: bool,
                 compute_input_grad: bool):
        self.num_nodes = num_nodes
        self.F_in = F_in
        self.F_out = F_out
        self.num_entries = num_entries
        self.val_dtype = val_dtype
        self.idx_dtype = idx_dtype
        self.do_bias = do_bias

        # Grad W = Grad C^T @ A^t @ X
        # = (A @ Grad C)^T @ X = Grad C^T @ (A^t @ X)
        weight_grad_subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = (A @ Grad Y) @ W = A @ (Grad Y @ W)
        input_grad_subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops

    def flops(self):
        return sum([op.flops() for op in self.subops])

    def min_memory(self):
        return sum([op.min_memory() for op in self.subops])