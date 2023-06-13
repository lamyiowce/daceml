# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
from copy import deepcopy as dc
import copy

import dace.library
import dace.sdfg.nodes
import numpy as np
from dace import SDFG, SDFGState
from dace import dtypes, memlet as mm, properties, data as dt, propagate_memlets_sdfg
from dace.libraries.blas.blas_helpers import (to_blastype, check_access, to_cublas_computetype)
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation

from examples.gnn_benchmark.backported.cusparse import cuSPARSE
from examples.gnn_benchmark.backported.intel_mkl import IntelMKLSparse


def _is_complex(dtype):
    if hasattr(dtype, "is_complex") and callable(dtype.is_complex):
        return dtype.is_complex()
    else:
        return dtype in [np.complex64, np.complex128]


def _cast_to_dtype_str(value, dtype: dace.dtypes.typeclass) -> str:
    if _is_complex(dtype) and _is_complex(type(value)):
        raise ValueError("Cannot use complex beta with non-complex array")

    if _is_complex(dtype):
        cast_value = complex(value)

        return "dace.{type}({real}, {imag})".format(
            type=dace.DTYPE_TO_TYPECLASS[dtype].to_string(),
            real=cast_value.real,
            imag=cast_value.imag,
        )
    else:
        return "dace.{}({})".format(dace.DTYPE_TO_TYPECLASS[dtype].to_string(), value)


def _get_csrmm_operands(node,
                        state,
                        sdfg,
                        name_lhs_rows="_a_rows",
                        name_lhs_cols="_a_cols",
                        name_lhs_vals="_a_vals",
                        name_rhs="_b",
                        name_out="_c"):
    """Returns the CSRMM input edges, arrays, and shape."""

    result = {}
    result[name_lhs_rows] = None
    result[name_lhs_cols] = None
    result[name_lhs_vals] = None
    result[name_rhs] = None
    result[name_out] = None

    for edge in state.all_edges(node):
        if edge.dst_conn in result.keys():
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed]
            res = edge, outer_array, size, strides
            result[edge.dst_conn] = res
        elif edge.src_conn == name_out:
            subset = dc(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_output_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed]
            result[edge.src_conn] = (edge, outer_array, size, strides)
    for name, res in result.items():
        if res is None:
            raise ValueError("Matrix multiplication connector "
                             "\"{}\" not found.".format(name))
    return result


@dace.library.expansion
class ExpandCSRMMMKL(ExpandTransformation):
    environments = [IntelMKLSparse]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        operands = _get_csrmm_operands(node, state, sdfg)
        arows = operands['_a_rows'][1]
        acols = operands['_a_cols'][1]
        avals = operands['_a_vals'][1]
        bdesc = operands['_b'][1]
        dtype = avals.dtype.base_type
        func = f"mkl_sparse_{to_blastype(dtype.type).lower()}"
        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'
        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        check_access(dtypes.ScheduleType.CPU_Multicore, arows, acols, avals, bdesc, cdesc)
        opt = {}

        opt['func'] = func

        if node.transA:
            opt['opA'] = 'SPARSE_OPERATION_TRANSPOSE'
        else:
            opt['opA'] = 'SPARSE_OPERATION_NON_TRANSPOSE'

        opt['layout'] = 'SPARSE_LAYOUT_ROW_MAJOR'

        code = ''
        if dtype in (dace.complex64, dace.complex128):
            code = f'''
            {dtype.ctype} alpha = {alpha};
            {dtype.ctype} beta = {beta};
            '''
            opt['alpha'] = '&alpha'
            opt['beta'] = '&beta'
        else:
            opt['alpha'] = alpha
            opt['beta'] = beta
        opt['nrows'] = cdesc.shape[0]
        opt['ncols'] = cdesc.shape[1]
        opt['arows'] = cdesc.shape[0]
        opt['acols'] = bdesc.shape[0]
        if node.transA:
            opt['arows'], opt['acols'] = opt['acols'], opt['arows']

        opt['ldb'] = opt['ncols']
        opt['ldc'] = opt['ncols']
        code += """
            sparse_matrix_t __csrA;
            {func}_create_csr(&__csrA, SPARSE_INDEX_BASE_ZERO, {arows}, {acols}, _a_rows, _a_rows + 1, _a_cols, _a_vals);
            struct matrix_descr __descrA;
            __descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
            __descrA.mode = SPARSE_FILL_MODE_UPPER;
            __descrA.diag = SPARSE_DIAG_NON_UNIT;
            {func}_mm({opA}, {alpha}, __csrA, __descrA, {layout}, _b, {ncols}, {ldb}, {beta}, _c, {ldc});
        """.format_map(opt)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )
        return tasklet


@dace.library.expansion
class ExpandCSRMMPure(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state: SDFGState, sdfg: SDFG):
        if node.transA:
            raise NotImplementedError("Pure expansion of CSRMM does not support transA")

        nsdfg = SDFG(node.label + "_nsdfg")

        operands = _get_csrmm_operands(node, state, sdfg)
        nstate = nsdfg.add_state("state", is_start_state=True)
        for name, desc in operands.items():
            desc = desc[1]

            if isinstance(desc, dt.View):
                ndesc = desc.as_array()
            else:
                ndesc = dc(desc)
            ndesc.lifetime = dtypes.AllocationLifetime.Scope
            ndesc.transient = False
            nsdfg.add_datadesc(name, ndesc)

        array_a_vals = nsdfg.arrays['_a_vals']
        array_a_rows = nsdfg.arrays['_a_rows']
        array_a_cols = nsdfg.arrays['_a_cols']
        array_b = nsdfg.arrays['_b']
        array_c = nsdfg.arrays['_c']

        a_val_node = nstate.add_access('_a_vals')
        a_row_node = nstate.add_access('_a_rows')
        a_col_node = nstate.add_access('_a_cols')
        b_node = nstate.add_access('_b')
        c_node = nstate.add_access('_c')

        if node.beta == 0.0:
            shape_c = operands['_c'][1].shape

            init_state = nsdfg.add_state_before(nstate, node.label + "_initstate")
            init_state.add_mapped_tasklet(
                'csrmm_init', {'_o%d' % i: '0:%s' % symstr(d)
                               for i, d in enumerate(shape_c)}, {},
                'out = 0', {'out': dace.Memlet.simple('_c', ','.join(
                    ['_o%d' % i for i in range(len(shape_c))]))},
                external_edges=True)
        elif node.beta == 1.0:
            # Simplify computation
            edges = state.in_edges_by_connector(node, "_cin")
            for edge in edges:
                state.remove_edge(edge)

                if state.in_degree(edge.src) == 0 and state.out_degree(edge.src) == 0:
                    state.remove_node(edge.src)

            node.remove_in_connector("_cin")
        else:
            init_state = nsdfg.add_state_before(nstate, node.label + "_initstate")

            cdesc = operands['_c'][1]
            cin_desc = dc(cdesc)
            nsdfg.add_datadesc('_cin', cin_desc)

            init_state.add_mapped_tasklet(
                'csrmm_init', {'_o%d' % i: '0:%s' % symstr(d)
                               for i, d in enumerate(cdesc.shape)},
                {'_in': dace.Memlet.simple('_cin', ','.join(
                    ['_o%d' % i for i in range(len(cdesc.shape))]))},
                f'_out = {node.beta} * _in',
                {'_out': dace.Memlet.simple('_c', ','.join(
                    ['_o%d' % i for i in range(len(cdesc.shape))]))},
                external_edges=True)

        # Multiplication map
        outer_map_entry, outer_map_exit = nstate.add_map("spmm_1", dict(
            i='0:' + str(array_a_rows.shape[0] - 1)))
        outer_map_entry.add_in_connector("IN__a_vals")
        outer_map_entry.add_in_connector("IN__a_cols")
        outer_map_entry.add_in_connector("IN__a_rows")
        outer_map_entry.add_in_connector("IN__b")
        outer_map_exit.add_out_connector("OUT__c")

        nstate.add_edge(a_val_node, None, outer_map_entry, "IN__a_vals",
                        mm.Memlet.from_array("_a_vals", array_a_vals))
        nstate.add_edge(a_col_node, None, outer_map_entry, "IN__a_cols",
                        mm.Memlet.from_array("_a_cols", array_a_cols))
        nstate.add_edge(a_row_node, None, outer_map_entry, "IN__a_rows",
                        mm.Memlet.from_array("_a_rows", array_a_rows))
        nstate.add_edge(b_node, None, outer_map_entry, "IN__b", mm.Memlet.from_array("_b", array_b))
        nstate.add_edge(outer_map_exit, "OUT__c", c_node, None, mm.Memlet.from_array("_c", array_c))

        outer_map_entry.add_out_connector("OUT__a_vals")
        outer_map_entry.add_out_connector("OUT__a_cols")
        outer_map_entry.add_out_connector("OUT__a_rows")
        outer_map_entry.add_out_connector("OUT__b")

        inner_map_entry, inner_map_exit = nstate.add_map("spmm_2",
                                                         dict(j="__map_19_b0:__map_19_e1"))
        inner_map_entry.add_in_connector("__map_19_b0")
        inner_map_entry.add_in_connector("__map_19_e1")
        nstate.add_edge(outer_map_entry, "OUT__a_rows", inner_map_entry, "__map_19_b0",
                        mm.Memlet("_a_rows[i]", data="_a_rows"))
        nstate.add_edge(outer_map_entry, "OUT__a_rows", inner_map_entry, "__map_19_e1",
                        mm.Memlet("_a_rows[i + 1]", data="_a_rows"))

        inner_map_entry.add_in_connector("IN_tmp_a_vals")
        nstate.add_edge(outer_map_entry, "OUT__a_vals", inner_map_entry, "IN_tmp_a_vals",
                        mm.Memlet.from_array("_a_vals", array_a_vals))

        inner_map_entry.add_in_connector("IN_tmp_a_cols")
        nstate.add_edge(outer_map_entry, "OUT__a_cols", inner_map_entry, "IN_tmp_a_cols",
                        mm.Memlet.from_array("_a_cols", array_a_cols))

        inner_map_entry.add_in_connector("IN_tmp_b")
        nstate.add_edge(outer_map_entry, "OUT__b", inner_map_entry, "IN_tmp_b",
                        mm.Memlet.from_array("_b", array_b))

        inner_map_exit.add_out_connector("OUT__c_1")
        outer_map_exit.add_in_connector("IN__c")
        nstate.add_edge(inner_map_exit, "OUT__c_1", outer_map_exit, "IN__c",
                        mm.Memlet(expr=f"_c[i, 0:{str(array_c.shape[1])}]"))

        inner_map_entry.add_out_connector("OUT_tmp_a_vals")
        inner_map_entry.add_out_connector("OUT_tmp_a_cols")
        inner_map_entry.add_out_connector("OUT_tmp_b")

        if node.transB:
            B_rows = array_b.shape[1]
            B_cols = array_b.shape[0]
        else:
            B_rows = array_b.shape[0]
            B_cols = array_b.shape[1]

        k_map_entry, k_map_exit = nstate.add_map("spmm_3", dict(k=f"0:{str(B_cols)}"))
        k_map_entry.add_in_connector("IN_tmp_a_vals_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_a_vals", k_map_entry, "IN_tmp_a_vals_1",
                        mm.Memlet.simple("_a_vals", "j"))

        k_map_entry.add_in_connector("IN_tmp_a_cols_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_a_cols", k_map_entry, "IN_tmp_a_cols_1",
                        mm.Memlet.simple("_a_cols", "j"))

        k_map_entry.add_in_connector("IN_tmp_b_1")
        nstate.add_edge(inner_map_entry, "OUT_tmp_b", k_map_entry, "IN_tmp_b_1",
                        mm.Memlet.from_array("_b", array_b))

        k_map_exit.add_out_connector("OUT__c_1")
        inner_map_exit.add_in_connector("IN__c_1")
        nstate.add_edge(k_map_exit, "OUT__c_1", inner_map_exit, "IN__c_1",
                        mm.Memlet(expr=f"_c[i, 0:{str(array_c.shape[1])}]"))

        k_map_entry.add_out_connector("OUT_tmp_a_cols_1")
        k_map_entry.add_out_connector("OUT_tmp_a_vals_1")
        k_map_entry.add_out_connector("OUT_tmp_b_1")

        tasklet_ind = nstate.add_tasklet("Indirection",
                                         inputs={
                                             "__ind_b": None,
                                             "index_a_cols_0": None
                                         },
                                         outputs={'lookup': None},
                                         code="lookup = __ind_b[index_a_cols_0]")
        nsdfg.add_scalar("_b_value", dtype=array_b.dtype, transient=True)
        nstate.add_edge(k_map_entry, "OUT_tmp_a_cols_1", tasklet_ind, "index_a_cols_0",
                        mm.Memlet.simple("_a_cols", "j"))
        nstate.add_edge(k_map_entry, "OUT_tmp_b_1", tasklet_ind, "__ind_b",
                        mm.Memlet.simple("_b",
                                         f"k, 0:{B_rows}" if node.transB else f"0:{B_rows}, k"))

        tasklet_mult = nstate.add_tasklet("spmm", {
            "__a": None,
            "__b": None
        }, {"__o": None},
                                          code=f"__o = {node.alpha} * (__a * __b)")
        nstate.add_edge(k_map_entry, "OUT_tmp_a_vals_1", tasklet_mult, "__a",
                        mm.Memlet.simple("_a_vals", "j"))
        nstate.add_edge(tasklet_ind, "lookup", tasklet_mult, "__b",
                        mm.Memlet.simple("_b_value", "0"))

        k_map_exit.add_in_connector("IN__c_1")
        nstate.add_edge(tasklet_mult, "__o", k_map_exit, "IN__c_1",
                        mm.Memlet.simple("_c", subset_str="i, k", wcr_str="lambda x, y: (x + y)"))

        nsdfg.validate()
        propagate_memlets_sdfg(nsdfg)

        return nsdfg


@dace.library.expansion
class ExpandCSRMMCuSPARSE(ExpandTransformation):
    environments = [cuSPARSE]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_csrmm_operands(node, state, sdfg)
        _, arows, arows_shape, _ = operands['_a_rows']
        acols = operands['_a_cols'][1]
        _, avals, avals_shape, _ = operands['_a_vals']
        _, bdesc, b_shape, _ = operands['_b']
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]
        # We need to use the shapes computed by _get_csrmm_operands, because it holds the
        # squeezed shapes. Otherwise, we would get the wrong shapes in case the input has
        # shape 2, 1, 4, for example.
        c_shape = operands['_c'][2]

        # If buffers are not on the GPU, copy them.
        needs_copy = any(
            desc.storage not in (dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)
            for desc in (arows, acols, avals, bdesc, cdesc))
        if needs_copy:
            print("!!!!!! Matrices not on GPU !!!!!!!")
            cpu_matrices = [(name, desc) for name, (_, desc, _, _) in operands.items() if
                            desc.storage not in (
                                dace.StorageType.GPU_Global, dace.StorageType.CPU_Pinned)]
            print(cpu_matrices)
            raise ValueError("matrices not on GPU: " + str(cpu_matrices))

        dtype = avals.dtype.base_type
        func = "cusparseSpMM"
        if dtype == dace.float16:
            cdtype = '__half'
            factort = 'Half'
        elif dtype == dace.float32:
            cdtype = 'float'
            factort = 'Float'
        elif dtype == dace.float64:
            cdtype = 'double'
            factort = 'Double'
        elif dtype == dace.complex64:
            cdtype = 'cuComplex'
            factort = 'Complex64'
        elif dtype == dace.complex128:
            cdtype = 'cuDoubleComplex'
            factort = 'Complex128'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        call_prefix = cuSPARSE.handle_setup_code(node)
        call_suffix = ''

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        else:
            alpha = f'{dtype.ctype}({node.alpha})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
        else:
            beta = f'{dtype.ctype}({node.beta})'

        # Set pointer mode to host
        call_prefix += f'''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        {dtype.ctype} alpha = {alpha};
        {dtype.ctype} beta = {beta};
        '''
        call_suffix += '''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);'''
        alpha = f'({cdtype} *)&alpha'
        beta = f'({cdtype} *)&beta'

        # Set up options for code formatting
        # opt = _get_codegen_gemm_opts(node, state, sdfg, adesc, bdesc, cdesc, alpha, beta, cdtype, func)

        # Get indices data type.
        idx_dtype = arows.dtype.base_type
        if idx_dtype not in [dace.int32, dace.int64]:
            raise ValueError(
                f"Unsupported index type: {idx_dtype} (only int32 and int64 supported).")

        opt = {}

        opt['arr_prefix'] = arr_prefix = ''
        if needs_copy:
            opt['arr_prefix'] = arr_prefix = '_conn'

        opt['func'] = func

        if node.transA:
            opt['opA'] = 'CUSPARSE_OPERATION_TRANSPOSE'
        else:
            opt['opA'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'

        if node.transB:
            opt['opB'] = 'CUSPARSE_OPERATION_TRANSPOSE'
        else:
            opt['opB'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'

        opt['layout'] = 'CUSPARSE_ORDER_ROW'

        opt['compute'] = f'CUDA_R_{to_cublas_computetype(dtype)}'
        idx_dtype_to_cusparse_dtype = {
            dace.int32: 'CUSPARSE_INDEX_32I',
            dace.int64: 'CUSPARSE_INDEX_64I'
        }
        opt['idx_dtype'] = idx_dtype_to_cusparse_dtype[idx_dtype]
        opt['handle'] = '__dace_cusparse_handle'

        opt['alpha'] = alpha
        opt['beta'] = beta

        # We use reverse indexing to support the batched case.
        opt['nrows'] = c_shape[-2]
        opt['ncols'] = c_shape[-1]
        opt['ldc'] = opt['ncols']

        opt['brows'] = b_shape[-2]
        opt['bcols'] = b_shape[-1]
        opt['ldb'] = opt['bcols']

        opt['arows'] = arows_shape[-1] - 1
        if node.transA:
            # Number of A cols is the number of rows in C.
            # A: M x N (to be transposed), B: M x K, C: N x K
            opt['acols'] = c_shape[-2]
        elif not node.transA and node.transB:
            # Number of A cols is the number of columns in B.
            # A: N x M, B: K x M (to be transposed), C: N x K
            opt['acols'] = b_shape[-1]
        elif not node.transA and not node.transB:
            # Number of A cols is the number of rows in B.
            # A: N x M, B: M x K, C: N x K
            opt['acols'] = b_shape[-2]


        opt['annz'] = avals_shape[-1]

        opt['algo'] = 'CUSPARSE_SPMM_CSR_ALG2'

        opt['num_batches'] = c_shape[0] if len(c_shape) > 2 else 1
        opt['avals_batch_stride'] = avals_shape[1] if len(avals_shape) > 1 else 0
        opt['b_batch_stride'] = b_shape[1] * b_shape[2] if len(b_shape) > 2 else 0
        opt['c_batch_stride'] = c_shape[1] * c_shape[2] if len(c_shape) > 2 else 0

        if opt['num_batches'] > 1:
            set_batches = """
                    // Set batch sizes and strides.
                    dace::sparse::CheckCusparseError( cusparseDnMatSetStridedBatch(matC, {num_batches}, {c_batch_stride}) );
                    dace::sparse::CheckCusparseError( cusparseDnMatSetStridedBatch(matB, {num_batches}, {b_batch_stride}) );
                    dace::sparse::CheckCusparseError( cusparseCsrSetStridedBatch(matA, {num_batches}, {avals_batch_stride}) );
                """
        else:
            set_batches = ""
        opt['set_batch_sizes_and_strides'] = set_batches.format_map(opt)

        call = """
            cusparseSpMatDescr_t matA;
            cusparseDnMatDescr_t matB, matC;
            // Create sparse matrix A in CSR format
            dace::sparse::CheckCusparseError( cusparseCreateCsr(&matA, {arows}, {acols}, {annz},
                                                {arr_prefix}_a_rows, {arr_prefix}_a_cols, {arr_prefix}_a_vals,
                                                {idx_dtype}, {idx_dtype},
                                                CUSPARSE_INDEX_BASE_ZERO, {compute}) );
            // Create dense matrix B
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matB, {brows}, {bcols}, {ldb}, {arr_prefix}_b,
                                                {compute}, {layout}) );
            // Create dense matrix C
            dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matC, {nrows}, {ncols}, {ldc}, {arr_prefix}_c,
                                                {compute}, {layout}) );
            
            {set_batch_sizes_and_strides}
            
            // Get the size of the additional buffer that's needed.
            size_t bufferSize;
            dace::sparse::CheckCusparseError( cusparseSpMM_bufferSize(
                                            {handle},
                                            {opA},
                                            {opB},
                                            {alpha}, matA, matB, {beta}, matC, {compute},
                                            {algo}, &bufferSize) );
            void* dBuffer = __state->cusparse_handle.Buffer(__dace_cuda_device, __dace_current_stream_id, bufferSize);

            // execute SpMM
            dace::sparse::CheckCusparseError( cusparseSpMM({handle},
                                            {opA},
                                            {opB},
                                            {alpha}, matA, matB, {beta}, matC, {compute},
                                            {algo}, dBuffer) );
            // destroy matrix/vector descriptors
            dace::sparse::CheckCusparseError( cusparseDestroySpMat(matA) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matB) );
            dace::sparse::CheckCusparseError( cusparseDestroyDnMat(matC) );
        """.format_map(opt)

        code = (call_prefix + call + call_suffix)
        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        # If buffers are not on the GPU, copy them
        if needs_copy:
            if node.beta != 0.0:
                from dace.transformation.interstate import GPUTransformSDFG

                nsdfg: dace.SDFG = ExpandCSRMMPure.expansion(node, state, sdfg)
                nsdfg.apply_transformations(GPUTransformSDFG)
                return nsdfg

            nsdfg = dace.SDFG('nested_gemm')
            copies = [('_a_rows', arows), ('_a_cols', acols), ('_a_vals', avals), ('_b', bdesc),
                      ('_c', cdesc)]
            for name, desc in copies:
                if isinstance(desc, dt.View):
                    dcopy = desc.as_array()
                else:
                    dcopy = dc(desc)
                dcopy.lifetime = dtypes.AllocationLifetime.Scope
                dcopy_gpu = dc(dcopy)
                dcopy.transient = False
                nsdfg.add_datadesc(name, dcopy)
                dcopy_gpu.transient = True
                dcopy_gpu.storage = dace.StorageType.GPU_Global
                nsdfg.add_datadesc(name + '_gpu', dcopy_gpu)
            nstate = nsdfg.add_state()
            ar = nstate.add_read('_a_rows')
            gar = nstate.add_access('_a_rows_gpu')
            ac = nstate.add_read('_a_cols')
            gac = nstate.add_access('_a_cols_gpu')
            av = nstate.add_read('_a_vals')
            gav = nstate.add_access('_a_vals_gpu')
            b = nstate.add_read('_b')
            gb = nstate.add_access('_b_gpu')
            c = nstate.add_write('_c')
            gc = nstate.add_access('_c_gpu')

            # Reset code and connectors
            tasklet.in_connectors = {"_conn" + k: None for k in tasklet.in_connectors}
            tasklet.out_connectors = {"_conn" + k: None for k in tasklet.out_connectors}

            nstate.add_node(tasklet)
            nstate.add_nedge(ar, gar, dace.Memlet.from_array('_a_rows', arows))
            nstate.add_nedge(ac, gac, dace.Memlet.from_array('_a_cols', acols))
            nstate.add_nedge(av, gav, dace.Memlet.from_array('_a_vals', avals))
            nstate.add_nedge(b, gb, dace.Memlet.from_array('_b', bdesc))

            nstate.add_edge(gar, None, tasklet, '_conn_a_rows',
                            dace.Memlet.from_array('_a_rows_gpu', arows))
            nstate.add_edge(gac, None, tasklet, '_conn_a_cols',
                            dace.Memlet.from_array('_a_cols_gpu', arows))
            nstate.add_edge(gav, None, tasklet, '_conn_a_vals',
                            dace.Memlet.from_array('_a_vals_gpu', arows))
            nstate.add_edge(gb, None, tasklet, '_conn_b', dace.Memlet.from_array('_b_gpu', bdesc))
            nstate.add_edge(tasklet, '_conn_c', gc, None, dace.Memlet.from_array('_c_gpu', cdesc))
            nstate.add_nedge(gc, c, dace.Memlet.from_array('_c', cdesc))

            return nsdfg
        # End of copy to GPU

        return tasklet


@dace.library.expansion
class ExpandCSRMMCpp(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_csrmm_operands(node, state, sdfg)
        # For getting the shape, we need to use the shape returned from _get_csrmm_operands
        # because then it is squeezed.
        _, avals, avals_shape, _ = operands['_a_vals']
        _, _, arows_shape, _ = operands['_a_rows']
        b_shape = operands['_b'][2]
        c_shape = operands['_c'][2]

        dtype = avals.dtype.base_type
        func = "cusparseSpMM"
        if dtype == dace.float32:
            cdtype = 'float'
        elif dtype == dace.float64:
            cdtype = 'double'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        # Deal with complex input constants
        if isinstance(node.alpha, complex):
            alpha = f'{dtype.ctype}({node.alpha.real}, {node.alpha.imag})'
        else:
            alpha = f'{dtype.ctype}({node.alpha})'
        if isinstance(node.beta, complex):
            beta = f'{dtype.ctype}({node.beta.real}, {node.beta.imag})'
        else:
            beta = f'{dtype.ctype}({node.beta})'

        # Set up options for code formatting
        opt = {}

        opt['dtype'] = cdtype

        opt['alpha'] = alpha
        opt['beta'] = beta

        opt['nrows'] = c_shape[-2]
        opt['ncols'] = c_shape[-1]

        opt['brows'] = b_shape[-2]
        opt['bcols'] = b_shape[-1]
        opt['arows'] = arows_shape[0] - 1

        opt['batch_size'] = c_shape[0] if len(c_shape) > 2 else 1
        opt['avals_batch_stride'] = avals_shape[1] if len(avals_shape) > 1 else 0
        opt['b_batch_stride'] = b_shape[1] * b_shape[2] if len(b_shape) > 2 else 0
        opt['c_batch_stride'] = c_shape[1] * c_shape[2] if len(c_shape) > 2 else 0

        if node.transA:
            code = """
                for (int b = 0; b < {batch_size}; b++) {{
                    for (int i = 0; i < {nrows}; i++) {{
                        for (int k = 0; k < {ncols}; k++) {{
                            _c[b * {c_batch_stride} + i * {ncols} + k] *= {beta};
                        }}
                    }}
                }}
                for (int b = 0; b < {batch_size}; b++) {{
                    for (int i = 0; i < {arows}; i++) {{
                        for (int k = 0; k < {ncols}; k++) {{
                            for (int j = _a_rows[i]; j < _a_rows[i + 1]; j++) {{
                                auto column = _a_cols[j];
                                {dtype} mult = {alpha} * _b[b * {b_batch_stride} + i * {bcols} + k] * _a_vals[b * {avals_batch_stride} + j];
                                _c[b * {c_batch_stride} + column * {ncols} + k] += mult;
                            }}
                        }}
                    }}
                }}
            """.format_map(opt)
        else:
            code = """
                for (int b = 0; b < {batch_size}; b++) {{
                    for (int i = 0; i < {nrows}; i++) {{
                        for (int k = 0; k < {ncols}; k++) {{
                            _c[b * {c_batch_stride} + i * {ncols} + k] *= {beta};
                        }}
                    }}
                }}
                for (int b = 0; b < {batch_size}; b++) {{
                    for (int i = 0; i < {arows}; i++) {{
                        for (int k = 0; k < {ncols}; k++) {{
                            for (int j = _a_rows[i]; j < _a_rows[i + 1]; j++) {{
                                auto column = _a_cols[j];
                                {dtype} mult = {alpha} * _b[b * {b_batch_stride} + column * {bcols} + k] * _a_vals[b * {avals_batch_stride} + j];
                                _c[b * {c_batch_stride} + i * {ncols} + k] += mult;
                            }}
                        }}
                    }}
                }}
            """.format_map(opt)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        return tasklet


@dace.library.node
class CSRMM(dace.sdfg.nodes.LibraryNode):
    """
    Executes alpha * (A @ B) + beta * C. C should be unidirectionally broadcastable (ONNX terminology) to A @ B.
    A is a sparse matrix in CSR format, while B is dense.
    """

    # Global properties
    implementations = {"cuSPARSE": ExpandCSRMMCuSPARSE,
                       "pure": ExpandCSRMMCuSPARSE} if os.environ.get('CUDA_VISIBLE_DEVICES',
                                                                      '') != '' else {
        "pure": ExpandCSRMMCpp}
    default_implementation = None

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0,
                               desc="A scalar which will be multiplied with C before adding C")

    def __init__(self, name, location=None, transA=False, transB=False, alpha=1, beta=0):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_rows", "_a_cols", "_a_vals", "_b", "_cin"}
                                 if beta != 0 and beta != 1.0 else {"_a_rows", "_a_cols", "_a_vals",
                                                                    "_b"}),
                         outputs={"_c"})
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) not in [4, 5]:
            raise ValueError("Expected 4 or 5 inputs to CSRMM")
        size4 = None

        sizes = {}
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            subset = copy.deepcopy(memlet.subset)
            subset.squeeze()
            sizes[dst_conn] = subset.size()

        for _, _, _, dst_conn, memlet in state.in_edges(self):
            if dst_conn == '_a_rows':
                subset = dc(memlet.subset)
                subset.squeeze()
                sizes['_a_rows'] = subset.size()
            if dst_conn == '_b':
                subset = dc(memlet.subset)
                subset.squeeze()
                sizes['_b']= subset.size()
            if dst_conn == '_cin':
                subset = dc(memlet.subset)
                subset.squeeze()
                sizes['_cin'] = subset.size()

        # Get output size.
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        sizes['_out'] = out_subset.size()

        # Function is symmetric, edge order does not matter
        if len(sizes['_b']) not in [2, 3]:
            raise ValueError("matrix-matrix product only supported on matrices or batched matrices.")

        A_rows = sizes['_a_rows'][0] - 1
        if self.transB:
            B_cols = sizes['_b'][0]
        else:
            B_cols = sizes['_b'][1]

        # if sizes['_a_rows'][1] != size1[0]:
        #     raise ValueError("Inputs to matrix-matrix product " "must agree in the k-dimension")

        if '_cin' in sizes and sizes['_cin'] != sizes['_out']:
            raise ValueError("Input C matrix must match output matrix.")
        if len(sizes['_out']) not in [2, 3]:
            raise ValueError("matrix-matrix product only supported on matrices or batched matrices.")
        if not self.transA and len(sizes['_out']) == 2 and list(sizes['_out']) != [A_rows, B_cols]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n "
                             "dimensions")

        if len(sizes['_a_cols']) != 1 or len(sizes['_a_rows']) != 1:
            raise NotImplementedError(f"A_rows and A_cols must be 1d, got {sizes['_a_cols']} "
                                      f"and {sizes['_a_rows']}. Batched SpMM supported only for the"
                                      f" same sparsity pattern for all batches.")

        # Check that all 3d matrices have the same batch dim.
        batch_dim_b = sizes['_b'][0] if len(sizes['_b']) == 3 else None
        batch_dim_out = sizes['_out'][0] if len(sizes['_out']) == 3 else None
        batch_dim_avals = sizes['_a_vals'][0] if len(sizes['_a_vals']) == 2 else None

        batch_dims = {'_b': batch_dim_b, '_out': batch_dim_out, '_a_vals': batch_dim_avals}
        batch_dims = {k: v for k, v in batch_dims.items() if v is not None}

        # We're using CUDA 11.4 which has a bug regarding the mode Ci = Ai @ B, so all matrices
        # require to be batched. (https://github.com/NVIDIA/CUDALibrarySamples/issues/81)
        # if len(batch_dims) not in [0, 3]:
        #     raise ValueError(
        #         "Either all or none of inputs and outputs to matrix-matrix product must be batched.")

        # If it's a batched op, then out has to have a batch dim and at least one of b and a has to
        # be batched as well.
        if len(batch_dims) > 0 and batch_dim_out is None:
            raise ValueError(
                "Output of matrix-matrix product must have a batch dimension if any of the inputs "
                "have a batch dimension.")
        if len(batch_dims) > 0 and batch_dim_b is None and batch_dim_avals is None:
            raise ValueError(
                "Either B or A_values must have a batch dimension in matrix-matrix product.")
        if len(batch_dims) > 0 and len(set(batch_dims.values())) != 1:
            raise ValueError(
                "Batch dimensions of B, A_values and output must match in matrix-matrix product. "
                f"Got {batch_dims}")
        if batch_dim_b is not None and batch_dim_out is not None and batch_dim_b != batch_dim_out:
            raise ValueError(
                f"Batch dimension of B and output must match in matrix-matrix product. Got "
                f"{batch_dim_b} and {batch_dim_out}")