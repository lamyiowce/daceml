import dace


class MeasurableOp:
    """Base class for operations that can be measured in terms of FLOPS and
         memory movement."""

    def flops(self):
        """Return the number of floating point operations."""
        raise NotImplementedError

    def min_memory(self):
        """Return the lower bound on memory movement in bytes."""
        raise NotImplementedError


class Matmul(MeasurableOp):
    # A @ B
    # A: N x M
    # B: M x F
    def __init__(self, N: int, M: int, F: int,
                 val_dtype: dace.dtypes.typeclass):
        self.N = N
        self.M = M
        self.F = F
        self.val_dtype = val_dtype

    def flops(self):
        return 2 * self.N * self.M * self.F

    def min_memory(self):
        return (
                self.N * self.M + self.M * self.F + self.N * self.F) * self.val_dtype.bytes


class Csrmm(MeasurableOp):
    # A @ B
    # A: N x M, CSR format, nnz non-zero entries
    # B: M x F
    def __init__(self, N: int, M: int, F: int, nnz: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.N = N
        self.M = M
        self.F = F
        self.nnz = nnz
        self.val_bytes = val_dtype.bytes
        self.idx_bytes = idx_dtype.bytes

    def flops(self):
        return 2 * self.nnz * self.F

    def min_memory(self):
        # Load: entry values, input matrix, output matrix.
        val_bytes = (
                            self.nnz + self.N * self.M + self.N * self.F) * self.val_bytes
        # Load: column indices, rowptrs.
        idx_bytes = (self.nnz + self.N + 1) * self.idx_bytes
        return val_bytes + idx_bytes


