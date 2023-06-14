import dataclasses

import dace
import pandas as pd

from examples.gnn_benchmark.report.measurable_layers import GCNConvCSR, \
    GCNConvCSRAdapt, BackwardGCNConvCSR, BackwardGCNConvCSRAdapt


@dataclasses.dataclass
class DatasetStats:
    name: str
    num_nodes: int
    num_edges: int
    num_features: int


CoraStats = DatasetStats(name='cora', num_nodes=2708, num_edges=10556,
                         num_features=1433)
ArxivStats = DatasetStats(name='arxiv', num_nodes=169343, num_edges=1166243,
                          num_features=128)


def compute_all(layer_name, hidden_sizes, datasets, layers, filename,
                **layer_kwargs):
    # model, impl_name, dataset, hidden_size, val_dtype, idx_dtype
    output = []
    for dataset in datasets:
        for hidden_size in hidden_sizes:
            for estimate_class in layers:
                layer = estimate_class(num_nodes=dataset.num_nodes,
                                       F_in=dataset.num_features,
                                       F_out=hidden_size,
                                       num_entries=dataset.num_edges,
                                       **layer_kwargs)
                output += [
                    (layer_name, layer.impl_name, dataset.name, hidden_size,
                     layer.flops(),
                     layer.flops() / 1024 ** 3,
                     layer.min_memory(),
                     layer.min_memory() / 1024 ** 3,
                     layer.flops() / layer.min_memory())]

    df = pd.DataFrame(output,
                      columns=['model', 'impl', 'dataset', 'hidden_size',
                               'flops', 'gigaflops', 'min_memory_bytes',
                               'min_memory_gb', 'op_intensity'])
    df.to_csv(filename, index=False)


def main():
    # Estimate the number of FLOPS and lower bound of memory movement for GAT
    # and GCN with typical sizes.

    hidden_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    datasets = [CoraStats, ArxivStats]
    gcn_layers = [GCNConvCSR, GCNConvCSRAdapt]
    backward_gcn_layers = [BackwardGCNConvCSR, BackwardGCNConvCSRAdapt]

    compute_all('gcn_single_layer', hidden_sizes, datasets, gcn_layers,
                'gcn_numbers.csv', val_dtype=dace.float32,
                idx_dtype=dace.int32,
                do_bias=True)

    compute_all('gcn_backward_single_layer', hidden_sizes, datasets,
                backward_gcn_layers, 'gcn_numbers_bwd.csv',
                val_dtype=dace.float32,
                idx_dtype=dace.int32,
                do_bias=True,
                compute_input_grad=True
                )


if __name__ == '__main__':
    main()
