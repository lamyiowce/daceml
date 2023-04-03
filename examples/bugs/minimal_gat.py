import dace
import numpy as np
import torch
from torch_sparse import SparseTensor

N = 4
heads = 2
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
def prog(node_features, rowptrs, columns, lin_srcDOTweight,
         att_src, att_dst, output):
    """
    node_features: input features, N x F
    rowptrs: rowptr, N+1
    columns: col, num_entries
    lin_srcDOTweight: H * F' x F
    att_srcDOT_weight: H x F
    output: N x H * F'
    """

    # Transform input features.
    features = dace.define_local((N, heads, num_out_features),
                                 dtype=np.float32)
    features_tmp = np.einsum('ij,kj->ik', node_features,
                             lin_srcDOTweight)
    features[:] = np.reshape(features_tmp,
                             (N, heads, num_out_features))
    # Compute node attention coefficients.
    alpha_src = np.sum(features * att_src, axis=-1)  # shape: N x H
    alpha_dst = np.sum(features * att_dst, axis=-1)  # N x H

    # Calculate attention weights.
    e = np.zeros((num_entries, heads), dtype=np.float32)
    softmax_sum = np.zeros((N, heads), dtype=np.float32)

    # TODO: Below loop can be flipped.
    for l in dace.map[0:N]:
        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
            # Calculating e_l->colv
            colv = columns[v]
            e_tmp = alpha_src[l] + alpha_dst[colv]
            # LeakyReLU
            e_tmp = np.maximum(0.1 * e_tmp, e_tmp)
            e_tmp = np.exp(e_tmp)
            e[v] = e_tmp
            softmax_sum[colv] += e_tmp

    # Softmax normalization.
    for j in dace.map[0:num_entries]:
        colj = columns[j]
        e[j] = e[j] / softmax_sum[colj]

    # Implementation with loop flattening.
    helper_row = dace.define_local((num_entries,), dtype=dace.int64)
    for l in dace.map[0:N]:
        for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
            helper_row[v] = l

    output[:] = 0
    for i in dace.map[0:num_entries]:
        colv = columns[i]
        b = helper_row[i]
        if heads == 1:
            output[colv] += e[i] * features[b]
        else:
            output[colv] += np.reshape(
                np.reshape(e[i], (heads, 1)) * features[b],
                (heads * num_out_features,))


def main():
    node_features = np.random.rand(N, num_in_features).astype(np.float32)

    adj_matrix = torch.tensor([[1., 0, 1, 0],
                               [1., 1, 1, 0],
                               [0., 1, 1, 1],
                               [0., 0, 1, 0]])
    adj = SparseTensor.from_dense(adj_matrix)
    rowptrs, columns, _ = adj.csr()
    weights = np.random.rand(heads * num_out_features, num_in_features).astype(
        np.float32)
    att_src = np.random.rand(heads, num_out_features).astype(np.float32)
    att_dst = np.random.rand(heads, num_out_features).astype(np.float32)
    output = np.zeros((N, heads * num_out_features), dtype=np.float32)
    expected_output = np.zeros((N, heads * num_out_features), dtype=np.float32)

    sdfg: dace.SDFG = prog.to_sdfg(node_features=node_features, rowptrs=rowptrs,
                                   columns=columns,
                                   lin_srcDOTweight=weights, att_src=att_src,
                                   att_dst=att_dst,
                                   output=output)

    # Transform the code to run on the GPU, while ensuring that the warp map
    # in the example runs within a single thread-block.
    if torch.cuda.is_available():
        sdfg.apply_gpu_transformations()
        dynamic_schedule(sdfg, exclude_loops = [
            # Below two have to be excluded because only one-dimensional
            # maps are supported in DaCe for dynamic block map schedule
            # (got 2).
            '_Div__map',
            '_Mult__map',
            # Below two have to be excluded, otherwise compile errors
            # occur (the generated code is incorrect).
            'assign_137_12_map',
            'outer_fused',
        ])
    sdfg(node_features=node_features, rowptrs=rowptrs, columns=columns,
         lin_srcDOTweight=weights, att_src=att_src, att_dst=att_dst,
         output=output)

    prog.f(node_features=node_features, rowptrs=rowptrs, columns=columns,
           lin_srcDOTweight=weights, att_src=att_src, att_dst=att_dst,
           output=expected_output)

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
