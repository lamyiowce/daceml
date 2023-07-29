from typing import Tuple

import dace
import numpy as np

WRITE_FACTOR = 2


class MeasurableOp:
    """Base class for operations that can be measured in terms of FLOPS and
         memory movement."""

    def flops(self):
        """Return the number of floating point operations."""
        raise NotImplementedError

    def min_memory(self):
        """Return the lower bound on memory movement in bytes."""
        raise NotImplementedError

    def op_intensity(self):
        """Return the operation intensity (FLOPS / byte)."""
        return self.flops() / self.min_memory()

    def basic_stats_str(self):
        gigflops = self.flops() / 10 ** 9
        memory_gigabytes = self.min_memory() / 10 ** 9
        return f'op_intensity={self.op_intensity():.3f} FLOP / B, flops={gigflops:.3f} GFLOP, min_memory={memory_gigabytes:.3f} GB'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.basic_stats_str()})'


class Matmul(MeasurableOp):
    # A @ B
    # A: N x M
    # B: M x F
    # Output: N x F
    def __init__(self, N: int, M: int, F: int,
                 val_dtype: dace.dtypes.typeclass):
        self.N = N
        self.M = M
        self.F = F
        self.val_dtype = val_dtype

    def flops(self):
        return 2 * self.N * self.M * self.F

    def min_memory(self):
        # Load A, B, store C (requires read and write).
        mem_count = self.N * self.M + self.M * self.F + WRITE_FACTOR * self.N * self.F
        return mem_count * self.val_dtype.bytes


class BatchedMatmul(MeasurableOp):
    # A @ B
    # A: B x N x M
    # B: B x M x F
    # Output: B x N x F

    def __init__(self, B: int, N: int, M: int, F: int,
                 val_dtype: dace.dtypes.typeclass):
        self.B = B
        self.matmul = Matmul(N, M, F, val_dtype)

    def flops(self):
        return self.B * self.matmul.flops()

    def min_memory(self):
        return self.B * self.matmul.min_memory()


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
        # Load: entry values, input matrix, output matrix (write requires read
        # and write).
        val_count = self.nnz + self.M * self.F + WRITE_FACTOR * self.N * self.F
        val_bytes = val_count * self.val_bytes
        # Load: column indices, rowptrs.
        idx_bytes = (self.nnz + self.N + 1) * self.idx_bytes
        return val_bytes + idx_bytes


class Cscmm(MeasurableOp):
    # A @ B
    # A: N x M, CSC format, nnz non-zero entries
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
        # Load: entry values, input matrix, output matrix (write requires read
        # and write).
        val_count = self.nnz + self.M * self.F + WRITE_FACTOR * self.N * self.F
        val_bytes = val_count * self.val_bytes
        # Load: column indices, rowptrs.
        idx_bytes = (self.nnz + self.M + 1) * self.idx_bytes
        return val_bytes + idx_bytes


class Coomm(MeasurableOp):
    # A @ B
    # A: N x M, COO format, nnz non-zero entries
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
        # Load: entry values, input matrix, output matrix (write requires read
        # and write).
        val_count = self.nnz + self.M * self.F + WRITE_FACTOR * self.N * self.F
        val_bytes = val_count * self.val_bytes
        # Load: column indices, row indices.
        idx_bytes = 2 * self.nnz * self.idx_bytes
        return val_bytes + idx_bytes


class SemibatchedCoomm(MeasurableOp):
    # C_i = A_i @ B_i, but graph structure is shared.
    def __init__(self, B: int, N: int, M: int, F: int, nnz: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.B = B
        self.N = N
        self.M = M
        self.F = F
        self.nnz = nnz
        self.val_bytes = val_dtype.bytes
        self.idx_bytes = idx_dtype.bytes
        self.single_coomm = Coomm(N, M, F, nnz, val_dtype, idx_dtype)

    def flops(self):
        return self.B * self.single_coomm.flops()

    def min_memory(self):
        # Load: entry values, input matrix, output matrix (write requires read
        # and write). B times.
        val_count = self.nnz + self.M * self.F + WRITE_FACTOR * self.N * self.F
        val_bytes = self.B * val_count * self.val_bytes
        # Load: column indices, row indices. Only once!
        idx_bytes = 2 * self.nnz * self.idx_bytes
        return val_bytes + idx_bytes


class Reduce(MeasurableOp):
    def __init__(self, shape: Tuple, axis: int,
                 val_dtype: dace.dtypes.typeclass):
        self.shape = shape
        self.axis = axis
        self.val_dtype = val_dtype

    def flops(self):
        # The reduction operation requires (M-1) additions for each axis entry,
        # where M is the size of the axis. We need to perform this for all
        # other axes.
        axis_size = self.shape[self.axis]
        other = np.prod(self.shape[:self.axis] + self.shape[self.axis + 1:])
        return (axis_size - 1) * other

    def min_memory(self):
        input_bytes = np.prod(self.shape) * self.val_dtype.bytes
        # We only need to load the whole output shape reduced along chosen axis.
        output_count = np.prod(self.shape[:self.axis] + self.shape[self.axis + 1:])
        output_bytes = WRITE_FACTOR * output_count * self.val_dtype.bytes
        return input_bytes + output_bytes


class AddBias(MeasurableOp):
    # A = A + b, where A is a tensor (e.g., N x M) and b is a vector (e. g., M).
    def __init__(self, shape: Tuple, axis: int,
                 val_dtype: dace.dtypes.typeclass):
        assert axis < len(shape)
        self.shape = shape
        self.axis = axis
        self.val_dtype = val_dtype

    def flops(self):
        # We need to add the bias to each entry of the input.
        op_count = np.prod(self.shape)
        return op_count

    def min_memory(self):
        input_bytes = np.prod(self.shape) * self.val_dtype.bytes
        bias_bytes = self.shape[self.axis] * self.val_dtype.bytes
        output_bytes = WRITE_FACTOR * np.prod(self.shape) * self.val_dtype.bytes
        return input_bytes + output_bytes + bias_bytes


class Permute(MeasurableOp):
    # B = np.permute(A, axes)
    def __init__(self, shape: Tuple, val_dtype: dace.dtypes.typeclass):
        self.shape = shape
        self.val_dtype = val_dtype

    def flops(self):
        return 0

    def min_memory(self):
        # We need to load the whole input and output tensors.
        elem_count = np.prod(self.shape)
        return (elem_count + WRITE_FACTOR * elem_count) * self.val_dtype.bytes


class PermuteAndAddBias(MeasurableOp):
    def __init__(self, shape: Tuple, axis: int,
                 val_dtype: dace.dtypes.typeclass):
        assert axis < len(shape)
        self.shape = shape
        self.axis = axis
        self.val_dtype = val_dtype

    def flops(self):
        # Add bias to each output element.
        return np.prod(self.shape)

    def min_memory(self):
        # Load: input (shape), bias (shape[axis]).
        # Write: output (permuted shape).
        elem_count = np.prod(self.shape)
        total_count = elem_count + WRITE_FACTOR * elem_count + self.shape[self.axis]
        return total_count * self.val_dtype.bytes


class CooSddmm(MeasurableOp):
    # A = np.sddmm(A, B, C), where A is a COO matrix N x N, and B is a dense N x M matrix,
    # and C is a dense M x N matrix.
    # for k, i in dace.map[0:K, 0:num_entries]:
    #     col = columns[i]
    #     row = rows[i]
    #     D[i] += e[i] * B[col, k] * F[row, k]
    def __init__(self, N: int, M: int, nnz: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.N = N
        self.M = M
        self.nnz = nnz
        self.val_bytes = val_dtype.bytes
        self.idx_bytes = idx_dtype.bytes

    def flops(self):
        return 3 * self.nnz * self.M

    def min_memory(self):
        # Load: B, C, each row/column is read multiple times, output matrix
        val_count = self.nnz * self.M * WRITE_FACTOR + 2 * self.nnz * self.M + self.nnz
        val_bytes = val_count * self.val_bytes
        # Load: column indices, row indices.
        idx_bytes = 2 * self.nnz * self.idx_bytes
        return val_bytes + idx_bytes


class FusedCooSpmmSddmm(MeasurableOp):
    # Fused COO SpMM and SDDMM. C = A @ B, where A is a COO matrix N x N,
    # B is a dense N x M matrix, and C is a dense N x M matrix.
    # D = A * (B @ C), where A is a COO matrix N x N, and B is a dense N x M matrix,
    # and C is a dense M x N matrix.
    # for k, i in dace.map[0:K, 0:num_entries]:
    #     col = columns[i]
    #     row = rows[i]
    #     D[i] += e[i] * B[col, k] * F[row, k]
    #     C[row, k] += e[i] * B[col, k]

    def __init__(self, N: int, M: int, nnz: int,
                 val_dtype: dace.dtypes.typeclass,
                 idx_dtype: dace.dtypes.typeclass):
        self.N = N
        self.M = M
        self.nnz = nnz
        self.val_bytes = val_dtype.bytes
        self.idx_bytes = idx_dtype.bytes

    def flops(self):
        sddmm_flops = 3 * self.nnz * self.M
        spmm_flops = 2 * self.nnz * self.M
        return sddmm_flops + spmm_flops

    def min_memory(self):
        # Load: B, C, F, each row/column is read multiple times
        dense_mem = self.nnz * self.M * WRITE_FACTOR + 2 * self.nnz * self.M
        # Load A, write D
        sparse_mem = self.nnz + WRITE_FACTOR * self.nnz
        val_bytes = (dense_mem + sparse_mem) * self.val_bytes
        # Load: column indices, row indices.
        idx_bytes = 2 * self.nnz * self.idx_bytes
        return val_bytes + idx_bytes
