import copy

import dace
from dace import memlet
from dace import properties
from dace.frontend.common import op_repository as oprepo
from dace.sdfg import SDFG, SDFGState
from dace.transformation import ExpandTransformation


def _get_operands(node,
                  state,
                  sdfg,
                  name_lhs_ellvalue="_a_ellvalue",
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
class ExpandBlockedEllpackMMCpp(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)

        operands = _get_operands(node, state, sdfg)
        aellvalue = operands['_a_ellvalue'][1]
        aellcolind = operands['_a_ellcolind'][1]
        avals = operands['_a_vals'][1]
        bdesc = operands['_b'][1]
        cdesc = sdfg.arrays[state.out_edges(node)[0].data.data]

        dtype = avals.dtype.base_type
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

        opt['num_ellcols'] = aellvalue.shape[1]
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
                            auto column = _a_ellcolind[j];
                            {dtype} mult = {alpha} * _b[column * {bcols} + k] * _a_ellvalues[i * {nrows} + j];
                            _c[i * {ncols} + k] += mult;
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
                            auto column = _a_ellcolind[j];
                            {dtype} mult = {alpha} * _b[i * {bcols} + k] * _a_ellvalues[i * {nrows} + j];
                            _c[column * {ncols} + k] += mult;
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
    implementations = {"pure": ExpandBlockedEllpackMMCpp}
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
    ellBlockSize = properties.Property(allow_none=False,
                                       dtype=int,
                                desc="Size of the ellpack block.")

    def __init__(self, name, ellBlockSize, location=None, transA=False, transB=False, alpha=1, beta=0):
        super().__init__(name,
                         location=location,
                         inputs=({"_a_ellvalue", "_a_ellcolind", "_b", "_cin"}
                                 if beta != 0 and beta != 1.0 else {"_a_ellvalue", "_a_ellcolind", "_b"}),
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

        if self.transA:
            A_cols, A_rows = sizes['_a_ellvalues']
        else:
            A_rows, A_cols = sizes['_a_ellvalues']

        B_rows, B_cols = sizes['_b']

        if A_cols != B_rows:
            raise ValueError("Inputs to matrix-matrix product must agree in the k-dimension")

        if sizes['_cin'] is not None and sizes['_cin'] != sizes['_out']:
            raise ValueError("Input C matrix must match output matrix.")
        if sizes['_out'] != [A_rows, B_cols]:
            raise ValueError("Output to matrix-matrix product must agree in the m and n "
                             "dimensions")


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
    # libnode.implementation = 'cuSPARSE' if torch.cuda.is_available() else 'pure'
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_ellcolind_in, None, libnode, '_a_ellcolind',
                   memlet.Memlet(A_ellcolind))
    state.add_edge(A_ellvalues_in, None, libnode, '_a_ellvalues',
                   memlet.Memlet(A_ellvalues))
    state.add_edge(B_in, None, libnode, '_b', memlet.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, memlet.Memlet(C))

    return []
