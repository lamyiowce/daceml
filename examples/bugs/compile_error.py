import dace
import numpy as np

N = 4
num_entries = 9
M = 3

np.random.seed(42)

def dynamic_schedule(sdfg, exclude_loops):
    """Change GPU sequential loops to dynamic."""
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
def prog(node_features, factor, rows, columns, output):
    """
    node_features: input features, N x F
    factor: num_entries
    rows: rowptr, num_entries
    columns: col, num_entries
    output: N x H * F'
    """
    output[:] = 0
    for i in dace.map[0:num_entries]:
        col = columns[i]
        row = rows[i]
        output[col] += factor[i] * node_features[row]


def main():
    node_features = np.random.rand(N, M).astype(np.float32)
    adj_matrix = np.array([[1., 0, 1, 0],
                           [1., 1, 1, 0],
                           [0., 1, 1, 1],
                           [0., 0, 1, 0]])
    rows, columns = adj_matrix.nonzero()
    rows = rows.copy()
    columns = columns.copy()
    factor = np.random.rand(num_entries).astype(np.float32)
    output = np.zeros((N, M), dtype=np.float32)
    expected_output = np.zeros((N, M), dtype=np.float32)

    sdfg: dace.SDFG = prog.to_sdfg(node_features=node_features, factor=factor,
                                   rows=rows,
                                   columns=columns,
                                   output=output)

    sdfg.apply_gpu_transformations()
    dynamic_schedule(sdfg, exclude_loops=[
        # The below map also doesn't allow to use the dynamic schedule because
        # only one-dimensional maps are supported in DaCe for dynamic block map
        # schedule (got 2), but that's a different issue.
        '_Mult__map',
        'assign_46_8_map',
    ])
    sdfg(node_features=node_features, factor=factor, rows=rows, columns=columns,
         output=output)

    prog.f(node_features=node_features, factor=factor, rows=rows,
           columns=columns,
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
