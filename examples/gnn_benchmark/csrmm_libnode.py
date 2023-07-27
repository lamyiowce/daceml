import dace
import torch
from dace import memlet
from dace.frontend.common import op_repository as oprepo
from dace.sdfg import SDFG, SDFGState

from examples.gnn_benchmark.backported.csrmm import CSRMM

# Number of rows and columns in A.
N = dace.symbol('N')
# Number of non-zero entries in A.
M = dace.symbol('M')
# Number of columns in C and B.
K = dace.symbol('K')


def csrmm(A_rowptrs,
          A_columns,
          A_values,
          B,
          C,
          alpha: float = 1.0,
          beta: float = 0.,
          transA: bool = False):
    C[:] = beta * C
    N = A_rowptrs.shape[0] - 1
    K = B.shape[1]
    if transA:
        for i, k in dace.map[0:N, 0:K]:
            for j in dace.map[A_rowptrs[i]:A_rowptrs[i + 1]]:
                # Below lines result in compile errors when enabling thread block dynamic scheduling.
                column = A_columns[j]
                mult = B[i, k] * A_values[j]
                C[column, k] += mult
    else:
        for i, k in dace.map[0:N, 0:K]:
            for j in dace.map[A_rowptrs[i]:A_rowptrs[i + 1]]:
                row = A_columns[j]
                mult = B[row, k] * A_values[j]
                C[i, k] += mult


@oprepo.replaces('examples.gnn_benchmark.csrmm_libnode.csrmm')
def csrmm_libnode(pv: 'ProgramVisitor',
                  sdfg: SDFG,
                  state: SDFGState,
                  A_rowptrs,
                  A_columns,
                  A_values,
                  B,
                  C, alpha=1., beta=0., transA=None):
    # Add nodes
    A_rowptrs_in, A_columns_in, A_values_in, B_in = (state.add_read(name) for
                                                     name in (
                                                         A_rowptrs, A_columns,
                                                         A_values, B))
    C_out = state.add_write(C)

    libnode = CSRMM('csrmm', transA=transA.item() if transA is not None else False, alpha=alpha, beta=beta)
    libnode.implementation = 'cuSPARSE' if torch.cuda.is_available() else 'pure'
    state.add_node(libnode)

    # Connect nodes
    state.add_edge(A_rowptrs_in, None, libnode, '_a_rows',
                   memlet.Memlet(A_rowptrs))
    state.add_edge(A_columns_in, None, libnode, '_a_cols',
                   memlet.Memlet(A_columns))
    state.add_edge(A_values_in, None, libnode, '_a_vals',
                   memlet.Memlet(A_values))
    state.add_edge(B_in, None, libnode, '_b', memlet.Memlet(B))
    state.add_edge(libnode, '_c', C_out, None, memlet.Memlet(C))

    return []
