import copy
import os
from typing import Optional

import dace
from dace import memlet
from dace import properties
from dace.frontend.common import op_repository as oprepo
from dace.libraries.blas.blas_helpers import to_cublas_computetype
from dace.sdfg import SDFG, SDFGState
from dace.transformation import ExpandTransformation

from examples.gnn_benchmark.backported.cusparse import cuSPARSE


def _get_operands(node,
                  state,
                  sdfg,
                  name_lhs_vals="_a_vals",
                  name_lhs_cols="_a_cols",
                  name_lhs_rows="_a_rows",
                  name_rhs="_b",
                  name_out="_c"):
    """Returns the COO input edges, arrays, and shape."""

    result = {}
    result[name_lhs_cols] = None
    result[name_lhs_rows] = None
    result[name_lhs_vals] = None
    result[name_rhs] = None
    result[name_out] = None

    for edge in state.all_edges(node):
        if edge.dst_conn in result.keys():
            subset = copy.deepcopy(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if
                       i in squeezed]
            res = edge, outer_array, size, strides
            result[edge.dst_conn] = res
        elif edge.src_conn == name_out:
            subset = copy.deepcopy(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(
                dace.sdfg.find_output_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if
                       i in squeezed]
            result[edge.src_conn] = (edge, outer_array, size, strides)
    for name, res in result.items():
        if res is None:
            raise ValueError("Matrix multiplication connector "
                             "\"{}\" not found.".format(name))
    return result


@dace.library.expansion
class ExpandCOOMMCuSPARSE(ExpandTransformation):
    environments = [cuSPARSE]

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_operands(node, state, sdfg)
        _, avalues, avalues_shape, _ = operands['_a_vals']
        _, acols, acols_shape, _ = operands['_a_cols']
        _, bdesc, b_shape, _ = operands['_b']
        _, cdesc, c_shape, _ = operands['_c']
        # cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        # Get values data type.
        dtype = avalues.dtype.base_type
        if dtype == dace.float16:
            cdtype = '__half'
        elif dtype == dace.float32:
            cdtype = 'float'
        elif dtype == dace.float64:
            cdtype = 'double'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        # Get indices data type.
        idx_dtype = acols.dtype.base_type
        if acols.dtype.base_type not in [dace.int32, dace.int64]:
            raise ValueError(
                f"Unsupported index type: {idx_dtype} (only int32 supported).")

        # Set up options for code formatting
        opt = {}

        opt['idx_dtype'] = f'CUSPARSE_INDEX_{"32" if idx_dtype == dace.int32 else "64"}I'
        opt['dtype'] = cdtype
        opt['handle'] = '__dace_cusparse_handle'

        alpha = f'({cdtype} *)&alpha'
        beta = f'({cdtype} *)&beta'
        opt['alpha'] = alpha
        opt['beta'] = beta

        if node.transA:
            opt['opA'] = 'CUSPARSE_OPERATION_TRANSPOSE'
        else:
            opt['opA'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'
        opt['opB'] = 'CUSPARSE_OPERATION_NON_TRANSPOSE'

        # Get sizes.
        opt['nnz'] = avalues_shape[0]
        opt['nrows'] = c_shape[0]
        opt['ncols'] = c_shape[1]
        opt['ldc'] = opt['ncols']

        if not node.transA:
            opt['arows'] = c_shape[0]
            opt['acols'] = b_shape[0]
        else:
            opt['arows'] = b_shape[0]
            opt['acols'] = c_shape[0]

        opt['brows'] = b_shape[0]
        opt['bcols'] = b_shape[1]
        opt['ldb'] = opt['bcols']

        opt['compute'] = f'CUDA_R_{to_cublas_computetype(dtype)}'
        opt['layout'] = 'CUSPARSE_ORDER_ROW'

        opt['algo'] = 'CUSPARSE_SPMM_COO_ALG4'

        call = """
                    cusparseSpMatDescr_t matA;
                    cusparseDnMatDescr_t matB, matC;
                    // Create sparse matrix A in CSR format
                    dace::sparse::CheckCusparseError( cusparseCreateCoo(
                        &matA, // cusparseSpMatDescr_t * spMatDescr,
                        {arows}, // int64_t rows,
                        {acols}, // int64_t cols,
                        {nnz}, // int64_t nnz,
                        _a_rows, // void * cooRowInd,
                        _a_cols, // void * cooColInd,
                        _a_vals, // void * cooValues,
                        {idx_dtype}, // cusparseIndexType_t cooIdxType,
                        CUSPARSE_INDEX_BASE_ZERO, // cusparseIndexBase_t idxBase,
                        {compute} // cudaDataType valueType
                        ));
                    // Create dense matrix B
                    dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matB, {brows}, {bcols}, {ldb}, _b,
                                                        {compute}, {layout}) );
                    // Create dense matrix C
                    dace::sparse::CheckCusparseError( cusparseCreateDnMat(&matC, {nrows}, {ncols}, {ldc}, _c,
                                                        {compute}, {layout}) );

                    // Get the size of the additional buffer that's needed.
                    // TODO: temp solution for dace_stream = nullptr.
  /*                  size_t bufferSize;
                    dace::sparse::CheckCusparseError( cusparseSpMM_bufferSize(
                                                    {handle},
                                                    {opA},
                                                    {opB},
                                                    {alpha}, matA, matB, {beta}, matC, {compute},
                                                    {algo}, &bufferSize)
                                                     );
                    void* dBuffer = __state->cusparse_handle.Buffer(__dace_cuda_device, 
                                                                    __dace_current_stream_id, 
                                                                    bufferSize);
*/ void* dBuffer = NULL;
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

        call_prefix = cuSPARSE.handle_setup_code(node)
        call_suffix = ''
        # Set pointer mode to host
        call_prefix += f'''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
                {dtype.ctype} alpha = {dtype.ctype}({node.alpha});
                {dtype.ctype} beta = {dtype.ctype}({node.beta});
                '''
        call_suffix += '''cusparseSetPointerMode(__dace_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);'''

        code = (call_prefix + call + call_suffix)

        tasklet = dace.sdfg.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code,
            language=dace.dtypes.Language.CPP,
        )

        return tasklet


@dace.library.expansion
class ExpandCOOMMCpp(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_operands(node, state, sdfg)
        _, avalues, avalues_shape, _ = operands['_a_vals']
        _, bdesc, b_shape, _ = operands['_b']
        _, cdesc, c_shape, _ = operands['_c']
        # cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        dtype = avalues.dtype.base_type
        if dtype == dace.float32:
            cdtype = 'float'
        elif dtype == dace.float64:
            cdtype = 'double'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        alpha = f'{dtype.ctype}({node.alpha})'
        beta = f'{dtype.ctype}({node.beta})'

        # Set up options for code formatting
        opt = {}

        opt['dtype'] = cdtype

        opt['alpha'] = alpha
        opt['beta'] = beta

        opt['num_entries'] = avalues_shape[0]
        opt['nrows'] = c_shape[0]
        opt['ncols'] = c_shape[1]

        opt['bcols'] = b_shape[1]

        if node.transA:
            code = """
                for (int i = 0; i < {nrows}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        _c[i * {ncols} + k] *= {beta};
                    }}
                }}
                for (int i = 0; i < {num_entries}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        auto column = _a_cols[i];
                        auto row = _a_rows[i];
                        {dtype} mult = {alpha} * _b[row * {bcols} + k] * _a_vals[i];
                        _c[column * {ncols} + k] += mult;
                    }}
                }}
            """.format_map(opt)
        else:
            code = """
                for (int i = 0; i < {nrows}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        _c[i * {ncols} + k] *= {beta};
                    }}
                }}
                for (int i = 0; i < {num_entries}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        auto column = _a_cols[i];
                        auto row = _a_rows[i];
                        {dtype} mult = {alpha} * _b[column * {bcols} + k] * _a_vals[i];
                        _c[row * {ncols} + k] += mult;
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
class COOMM(dace.sdfg.nodes.LibraryNode):
    """
    Executes alpha * (A @ B) + beta * C. C should be unidirectionally broadcastable (ONNX terminology) to A @ B.
    A is a sparse matrix in CSR format, while B is dense.
    """

    # Global properties
    implementations = {"cuSPARSE": ExpandCOOMMCuSPARSE} if os.environ.get(
        'CUDA_VISIBLE_DEVICES', '') != '' else {
        "pure": ExpandCOOMMCpp}
    default_implementation = None

    # Object fields
    transA = properties.Property(dtype=bool,
                                 desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool,
                                 desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1.,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0.,
                               desc="A scalar which will be multiplied with C before adding C")

    def __init__(self, name, location=None, transA=False, transB=False,
                 alpha=1., beta=0.):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_vals", "_a_rows", "_a_cols", "_b", "_cin"}
                                 if beta != 0 and beta != 1.0 else {"_a_vals",
                                                                    "_a_rows",
                                                                    "_a_cols",
                                                                    "_b"}),
                         outputs={"_c"})
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 4:
            raise ValueError("Expected 4 inputs to COOMM.")

        # Get sizes of all memlets.
        sizes = {'_cin': None}
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            subset = copy.deepcopy(memlet.subset)
            subset.squeeze()
            sizes[dst_conn] = subset.size()

        # Get output size.
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError(
                "Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        sizes['_out'] = out_subset.size()

        # Check all are 2d matrices.
        wrong_size_lens = {k: len(v) for k, v in sizes.items() if
                           v is not None and len(v) != 2 and '_a_' not in k}
        if len(wrong_size_lens) > 0:
            raise ValueError(
                f"matrix-matrix product only supported on matrices. Got: {wrong_size_lens}")

        B_rows, B_cols = sizes['_b']

        if sizes['_cin'] is not None and sizes['_cin'] != sizes['_out']:
            raise ValueError("Input C matrix must match output matrix.")

        if not self.transA:
            if sizes['_out'][1] != B_cols:
                raise ValueError(
                    "Output to matrix-matrix product must agree in the m and n "
                    "dimensions")


# Number of rows and columns in A.
N = dace.symbol('N')
# Number of non-zero entries in A.
M = dace.symbol('M')
# Number of columns in C and B.
K = dace.symbol('K')


def coomm(
        A_rows,
        A_cols,
        A_vals,
        B,
        C,
        transA: bool = False,
        alpha: float = 1.0,
        beta: float = 0.):
    pass


@oprepo.replaces('examples.gnn_benchmark.sparse_mm.coomm.coomm')
def coomm_libnode(pv: 'ProgramVisitor',
                  sdfg: SDFG,
                  state: SDFGState,
                  A_rows,
                  A_cols,
                  A_vals,
                  B,
                  C,
                  alpha: float = 1.,
                  beta: float = 0.,
                  transA: Optional[bool] = None):
    A_cols_in, A_rows_in, A_vals_in, B_in = (state.add_read(name) for
                                             name in (
                                                 A_cols, A_rows, A_vals, B))
    C_out = state.add_write(C)

    libnode = COOMM('coomm',
                    transA=transA.item() if transA is not None else False,
                    alpha=alpha,
                    beta=beta)
    libnode.implementation = "cuSPARSE" if os.environ.get(
        'CUDA_VISIBLE_DEVICES', '') != '' else "pure"
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_cols_in, None, libnode, '_a_cols',
                   memlet.Memlet(A_cols))
    state.add_edge(A_rows_in, None, libnode, '_a_rows',
                   memlet.Memlet(A_rows))
    state.add_edge(A_vals_in, None, libnode, '_a_vals',
                   memlet.Memlet(A_vals))
    state.add_edge(B_in, None, libnode, '_b', memlet.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, memlet.Memlet(C))

    return []
