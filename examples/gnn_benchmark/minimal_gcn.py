import dace
import numpy as np
import torch
from torch_sparse import SparseTensor

from examples.gnn_benchmark import csrmm_libnode

N = 4
num_entries = 9
num_out_features = 3
num_in_features = 6

np.random.seed(2137)


def dynamic_schedule(sdfg, exclude_loops):
    exclude_loops = {name: 0 for name in exclude_loops} or {}
    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential \
                and len(node[0].map.params):
            if node[0].label not in exclude_loops:
                print("Changing schedule to TB dynamic: ", node[0].map)
                node[0].schedule = dace.ScheduleType.GPU_ThreadBlock_Dynamic
            else:
                exclude_loops[node[0].label] += 1
                print("Keeping schedule sequential for ", node[0].map)

    not_excluded = [
        name for name, count in exclude_loops.items() if count == 0
    ]
    if not_excluded:
        print(
            "Following loops were marked as excluded from thread-block dynamic "
            "scheduling but were not found in the SDFG: %s", not_excluded)


@dace.program
def prog(node_features, rowptrs, columns, edge_vals, linDOTweight,
         bias, output):
    """
    node_features: input features, N x M
    rowptrs: row pointers (CSR format), N+1
    columns: col, num_entries
    edge_vals: values, num_entries
    linDOTweight: F x M
    output: N x F
    """
    features = dace.define_local((N, num_out_features), dtype=dace.float32)
    features[:] = np.einsum('ij,kj->ik', node_features, linDOTweight)
    for i, j in dace.map[0:N, 0:num_out_features]:
        output[i, j] = bias[j]
    csrmm_libnode.csrmm(rowptrs, columns, edge_vals, features, output, beta=1.0, transA=True)


def main():
    node_features = np.random.rand(N, num_in_features).astype(np.float32)

    adj_matrix = torch.tensor([[1., 0, 1, 0],
                               [1., 1, 1, 0],
                               [0., 1, 1, 1],
                               [0., 0, 1, 0]])
    adj = SparseTensor.from_dense(adj_matrix)
    rowptrs, columns, edge_vals = adj.csr()
    weights = np.random.rand(num_out_features, num_in_features).astype(
        np.float32)
    bias = np.random.rand(num_out_features).astype(np.float32)
    output = np.zeros((N, num_out_features), dtype=np.float32)
    expected_output = np.zeros((N, num_out_features), dtype=np.float32)

    sdfg: dace.SDFG = prog.to_sdfg(node_features=node_features,
                                   rowptrs=rowptrs,
                                   columns=columns,
                                   linDOTweight=weights,
                                   edge_vals=edge_vals,
                                   bias=bias,
                                   output=output)

    if torch.cuda.is_available():
        sdfg.apply_gpu_transformations()
    sdfg(node_features=node_features,
         rowptrs=rowptrs,
         columns=columns,
         linDOTweight=weights,
         edge_vals=edge_vals,
         bias=bias,
         output=output)

    prog.f(node_features=node_features,
           rowptrs=rowptrs,
           columns=columns,
           linDOTweight=weights,
           edge_vals=edge_vals,
           bias=bias,
           output=output)

    if np.allclose(output, expected_output):
        print("\n==== Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print("\n*↯*↯*↯* INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ *↯*↯*↯*")

    print("Actual output:")
    print(output)
    print("Expected output:")
    print(expected_output)


if __name__ == '__main__':
    main()
