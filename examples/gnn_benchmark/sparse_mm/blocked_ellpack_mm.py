import copy
import os

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
                  name_lhs_ellvalue="_a_ellvalues",
                  name_lhs_ellcolind="_a_ellcolind",
                  name_rhs="_b",
                  name_out="_c"):
    """Returns the Blocked Ellpack input edges, arrays, and shape."""

    result = {}
    result[name_lhs_ellcolind] = None
    result[name_lhs_ellvalue] = None
    result[name_rhs] = None
    result[name_out] = None

    for edge in state.all_edges(node):
        if edge.dst_conn in result.keys():
            subset = copy.deepcopy(edge.data.subset)
            squeezed = subset.squeeze()
            size = subset.size()
            outer_array = sdfg.data(dace.sdfg.find_input_arraynode(state, edge).data)
            strides = [s for i, s in enumerate(outer_array.strides) if i in squeezed]
            res = edge, outer_array, size, strides
            result[edge.dst_conn] = res
        elif edge.src_conn == name_out:
            subset = copy.deepcopy(edge.data.subset)
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
class ExpandBlockedEllpackMMCuSPARSE(ExpandTransformation):
    environments = [cuSPARSE]

    @staticmethod
    def expansion(node, state, sdfg):
        assert node.transA == False, "A cannot be transposed"
        node.validate(sdfg, state)

        operands = _get_operands(node, state, sdfg)
        aellvalues = operands['_a_ellvalues'][1]
        aellcolind = operands['_a_ellcolind'][1]
        bdesc = operands['_b'][1]
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        # Get values data type.
        dtype = aellvalues.dtype.base_type
        if dtype == dace.float32:
            cdtype = 'float'
        elif dtype == dace.float64:
            cdtype = 'double'
        else:
            raise ValueError("Unsupported type: " + str(dtype))

        # Get indices data type.
        idx_dtype = aellcolind.dtype.base_type
        if aellcolind.dtype.base_type != dace.int32:
            raise ValueError(f"Unsupported index type: {idx_dtype} (only int32 supported).")

        # Set up options for code formatting
        opt = {}

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
        opt['num_ellcols'] = aellvalues.shape[1]
        opt['nrows'] = cdesc.shape[0]
        opt['ncols'] = cdesc.shape[1]
        opt['ldc'] = opt['ncols']

        opt['arows'] = cdesc.shape[0]
        opt['acols'] = bdesc.shape[0]

        opt['brows'] = bdesc.shape[0]
        opt['bcols'] = bdesc.shape[1]
        opt['ldb'] = opt['bcols']

        opt['compute'] = f'CUDA_R_{to_cublas_computetype(dtype)}'
        opt['layout'] = 'CUSPARSE_ORDER_ROW'

        opt['ellBlockSize'] = node.ellBlockSize
        assert node.ellBlockSize == 1, "Other ell block sizes not supproted"

        call = """
                    cusparseSpMatDescr_t matA;
                    cusparseDnMatDescr_t matB, matC;
                    // Create sparse matrix A in CSR format
                    dace::sparse::CheckCusparseError( cusparseCreateBlockedEll(
                        &matA, // cusparseSpMatDescr_t * spMatDescr,
                        {arows}, // int64_t rows,
                        {acols}, // int64_t cols,
                        {ellBlockSize}, // int64_t ellBlockSize,
                        {num_ellcols}, // int64_t ellCols,
                        _a_ellcolind, // void * ellColInd,
                        _a_ellvalues, // void * ellValue,
                        CUSPARSE_INDEX_32I, // cusparseIndexType_t ellIdxType,
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
                    size_t bufferSize;
                    dace::sparse::CheckCusparseError( cusparseSpMM_bufferSize(
                                                    {handle},
                                                    {opA},
                                                    {opB},
                                                    {alpha}, matA, matB, {beta}, matC, {compute},
                                                    CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
                    void* dBuffer = __state->cusparse_handle.Buffer(__dace_cuda_device, 
                                                                    __dace_current_stream_id, 
                                                                    bufferSize);

                    // execute SpMM
                    dace::sparse::CheckCusparseError( cusparseSpMM({handle},
                                                    {opA},
                                                    {opB},
                                                    {alpha}, matA, matB, {beta}, matC, {compute},
                                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
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
class ExpandBlockedEllpackMMCpp(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_operands(node, state, sdfg)
        aellvalues = operands['_a_ellvalues'][1]
        aellcolind = operands['_a_ellcolind'][1]
        bdesc = operands['_b'][1]
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        dtype = aellvalues.dtype.base_type
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

        opt['num_ellcols'] = aellvalues.shape[1]
        opt['nrows'] = cdesc.shape[0]
        opt['ncols'] = cdesc.shape[1]

        opt['bcols'] = bdesc.shape[1]

        assert node.ellBlockSize == 1, "Other ell block sizes not supproted"

        if node.transA:
            code = """
                for (int i = 0; i < {nrows}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        _c[i * {ncols} + k] *= {beta};
                    }}
                }}
                for (int i = 0; i < {nrows}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        for (int j = 0; j < {num_ellcols}; j++) {{
                            auto column = _a_ellcolind[i * {num_ellcols} + j];
                            {dtype} mult = {alpha} * _b[i * {bcols} + k] * _a_ellvalues[i * {num_ellcols} + j];
                            _c[column * {ncols} + k] += mult;
                        }}
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
                for (int i = 0; i < {nrows}; i++) {{
                    for (int k = 0; k < {ncols}; k++) {{
                        for (int j = 0; j < {num_ellcols}; j++) {{
                            auto column = _a_ellcolind[i * {num_ellcols} + j];
                            {dtype} mult = {alpha} * _b[column * {bcols} + k] * _a_ellvalues[i * {num_ellcols} + j];
                            _c[i * {ncols} + k] += mult;
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
class BlockedEllpackMM(dace.sdfg.nodes.LibraryNode):
    """
    Executes alpha * (A @ B) + beta * C. C should be unidirectionally broadcastable (ONNX terminology) to A @ B.
    A is a sparse matrix in CSR format, while B is dense.
    """

    # Global properties
    implementations = {"cuSPARSE": ExpandBlockedEllpackMMCuSPARSE} if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else {
        "pure": ExpandBlockedEllpackMMCpp}
    default_implementation = None

    # Object fields
    transA = properties.Property(dtype=bool, desc="Whether to transpose A before multiplying")
    transB = properties.Property(dtype=bool, desc="Whether to transpose B before multiplying")
    alpha = properties.Property(allow_none=False,
                                default=1.,
                                desc="A scalar which will be multiplied with A @ B before adding C")
    beta = properties.Property(allow_none=False,
                               default=0.,
                               desc="A scalar which will be multiplied with C before adding C")
    ellBlockSize = properties.Property(allow_none=False,
                                       dtype=int,
                                       desc="Size of the ellpack block.")

    def __init__(self, name, ellBlockSize, location=None, transA=False, transB=False, alpha=1., beta=0.):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_ellvalues", "_a_ellcolind", "_b", "_cin"}
                                 if beta != 0 and beta != 1.0 else {"_a_ellvalues", "_a_ellcolind", "_b"}),
                         outputs={"_c"})
        self.ellBlockSize = ellBlockSize
        self.transA = transA
        self.transB = transB
        self.alpha = alpha
        self.beta = beta

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 3:
            raise ValueError("Expected 3 inputs to Blocked Ellpack.")

        # Get sizes of all memlets.
        sizes = {'_cin': None}
        for _, _, _, dst_conn, memlet in state.in_edges(self):
            subset = copy.deepcopy(memlet.subset)
            subset.squeeze()
            sizes[dst_conn] = subset.size()

        # Get output size.
        out_edges = state.out_edges(self)
        if len(out_edges) != 1:
            raise ValueError("Expected exactly one output from matrix-matrix product")
        out_memlet = out_edges[0].data
        out_subset = copy.deepcopy(out_memlet.subset)
        out_subset.squeeze()
        sizes['_out'] = out_subset.size()

        # Check all are 2d matrices.
        wrong_size_lens = {k: len(v) for k, v in sizes.items() if v is not None and len(v) != 2}
        if len(wrong_size_lens) > 0:
            raise ValueError(f"matrix-matrix product only supported on matrices. Got: {wrong_size_lens}")

        A_ellrows, _ = sizes['_a_ellvalues']

        B_rows, B_cols = sizes['_b']

        if sizes['_cin'] is not None and sizes['_cin'] != sizes['_out']:
            raise ValueError("Input C matrix must match output matrix.")

        if not self.transA:
            if sizes['_out'] != [A_ellrows, B_cols]:
                raise ValueError("Output to matrix-matrix product must agree in the m and n "
                                 "dimensions")
        else:
            if A_ellrows != B_rows:
                raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")


# Number of rows and columns in A.
N = dace.symbol('N')
# Number of non-zero entries in A.
M = dace.symbol('M')
# Number of columns in C and B.
K = dace.symbol('K')


def blocked_ellpack_mm(
        A_ellcolind,
        A_ellvalues,
        ellBlockSize,
        B,
        C,
        alpha: float = 1.0,
        beta: float = 0.,
        transA: bool = False):
    C[:] = beta * C
    N = A_ellvalues.shape[0]
    num_ell_cols = A_ellvalues.shape[1]
    K = B.shape[1]
    assert ellBlockSize == 1

    if not transA:
        for i, k in dace.map[0:N, 0:K]:
            for j in dace.map[0:num_ell_cols]:
                # i: row idx.
                column = A_ellcolind[i, j]
                mult = alpha * A_ellvalues[i, j] * B[column, k]
                C[i, k] += mult
    else:
        for i, k in dace.map[0:N, 0:K]:
            for j in dace.map[0:num_ell_cols]:
                # i: row idx.
                column = A_ellcolind[i, j]
                mult = alpha * A_ellvalues[i, j] * B[i, k]
                C[column, k] += mult


@oprepo.replaces('examples.gnn_benchmark.sparse_mm.blocked_ellpack_mm.blocked_ellpack_mm')
def blocked_ellpack_mm_libnode(pv: 'ProgramVisitor',
                               sdfg: SDFG,
                               state: SDFGState,
                               A_ellcolind,
                               A_ellvalues,
                               ellBlockSize: int,
                               B,
                               C,
                               alpha=1.,
                               beta=0.,
                               transA=None):
    assert ellBlockSize == 1
    # Add nodes
    A_ellcolind_in, A_ellvalues_in, B_in = (state.add_read(name) for
                                            name in (
                                                A_ellcolind, A_ellvalues, B))
    C_out = state.add_write(C)

    libnode = BlockedEllpackMM('blocked_ellpack_mm', ellBlockSize=1,
                               transA=transA.item() if transA is not None else False, alpha=alpha,
                               beta=beta)
    libnode.implementation = "cuSPARSE" if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else "pure"
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_ellcolind_in, None, libnode, '_a_ellcolind',
                   memlet.Memlet(A_ellcolind))
    state.add_edge(A_ellvalues_in, None, libnode, '_a_ellvalues',
                   memlet.Memlet(A_ellvalues))
    state.add_edge(B_in, None, libnode, '_b', memlet.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, memlet.Memlet(C))

    return []
