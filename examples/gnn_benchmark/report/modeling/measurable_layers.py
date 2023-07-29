from typing import List

import dace

from examples.gnn_benchmark.report.modeling.measurable_ops import MeasurableOp, Csrmm, \
    Matmul, Reduce, AddBias, Coomm, Cscmm


class CompositeOp(MeasurableOp):
    subops: List[MeasurableOp] = []
    def __init__(self, oplist: List[MeasurableOp]):
        self.subops = oplist

    def flops(self):
        return sum([op.flops() for op in self.subops])

    def min_memory(self):
        return sum([op.min_memory() for op in self.subops])

    def __repr__(self):
        subops_str = 'subops=[\n'
        for op in self.subops:
            subops_str += f'    {repr(op)},\n'
        subops_str += ']'
        return f'{self.__class__.__name__}({self.basic_stats_str()}, \n{subops_str})'


class GCNConvCSR(CompositeOp):
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

        self.subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class GCNConvCSRAdapt(CompositeOp):
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
        self.subops = [
            Csrmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class BackwardGCNConvCSR(CompositeOp):
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


class BackwardGCNConvCSRAdapt(CompositeOp):
    impl_name = 'csr_adapt'

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


class GCNConvCOO(CompositeOp):
    """Estimations for GCNConvCSR node."""
    impl_name = 'coo'

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

        self.subops = [
            Coomm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class GCNConvCOOAdapt(CompositeOp):
    """Estimations for GCNConvCSR node."""
    impl_name = 'coo_adapt'

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

        self.subops = [
            Coomm(N=num_nodes, M=num_nodes, F=min(F_out, F_in), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class BackwardGCNConvCOO(CompositeOp):
    impl_name = 'coo'

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
            Coomm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = A @ Grad G @ W
        input_grad_subops = [
            Coomm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops


class BackwardGCNConvCOOAdapt(CompositeOp):
    impl_name = 'coo_adapt'

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
            Coomm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = (A @ Grad Y) @ W = A @ (Grad Y @ W)
        input_grad_subops = [
            Coomm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops


class GCNConvCSC(CompositeOp):
    """Estimations for GCNConvCSC node."""
    impl_name = 'csc'

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

        self.subops = [
            Cscmm(N=num_nodes, M=num_nodes, F=F_out, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class GCNConvCSCAdapt(CompositeOp):
    """Estimations for GCNConvCSCAdapt node."""
    impl_name = 'csc_adapt'

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
        self.subops = [
            Cscmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_in, F=F_out, val_dtype=val_dtype),
            AddBias(shape=(num_nodes, F_out), axis=1, val_dtype=val_dtype)
        ]


class BackwardGCNConvCSC(CompositeOp):
    impl_name = 'csc'

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
            Cscmm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = A @ Grad G @ W
        input_grad_subops = [
            Cscmm(N=num_nodes, M=num_nodes, F=F_in, nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops


class BackwardGCNConvCSCAdapt(CompositeOp):
    impl_name = 'csc_adapt'

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
            Cscmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=F_out, M=num_nodes, F=F_in, val_dtype=val_dtype)
        ]

        # Grad X = (A @ Grad Y) @ W = A @ (Grad Y @ W)
        input_grad_subops = [
            Cscmm(N=num_nodes, M=num_nodes, F=min(F_in, F_out), nnz=num_entries,
                  val_dtype=val_dtype, idx_dtype=idx_dtype),
            Matmul(N=num_nodes, M=F_out, F=F_in, val_dtype=val_dtype)
        ] if compute_input_grad else []

        bias_grad_subops = [
            Reduce(shape=(num_nodes, F_out), axis=0, val_dtype=val_dtype)
        ] if do_bias else []

        self.subops = weight_grad_subops + input_grad_subops + bias_grad_subops


