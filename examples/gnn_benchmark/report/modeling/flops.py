import dataclasses

import dace
import pandas as pd

from examples.gnn_benchmark.report.modeling import measurable_ops, measurable_layers
from examples.gnn_benchmark.report.modeling.measurable_gat import GATConvCSR, GATConvCOO
from examples.gnn_benchmark.report.modeling.measurable_layers import GCNConvCSR, \
    GCNConvCSRAdapt, BackwardGCNConvCSR, BackwardGCNConvCSRAdapt, GCNConvCOO, GCNConvCOOAdapt, \
    BackwardGCNConvCOO, BackwardGCNConvCOOAdapt, GCNConvCSC, GCNConvCSCAdapt, \
    BackwardGCNConvCSCAdapt, BackwardGCNConvCSC


@dataclasses.dataclass
class DatasetStats:
    name: str
    num_nodes: int
    num_edges: int
    num_features: int
    max_row_size: int


CoraStats = DatasetStats(name='cora', num_nodes=2708, num_edges=10556,
                         num_features=1433, max_row_size=168)
ArxivStats = DatasetStats(name='arxiv', num_nodes=169343, num_edges=1166243,
                          num_features=128, max_row_size=436)


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
                print(f"{dataset.name}, {hidden_size}: {layer}")
                if hasattr('layer', 'impl_name'):
                    op_name = layer.impl_name
                else:
                    op_name = layer.__class__.__name__

                output += [
                    (layer_name, op_name, dataset.name, hidden_size,
                     layer.flops(),
                     layer.flops() / 1024 ** 3,
                     layer.min_memory(),
                     layer.min_memory() / 1024 ** 3,
                     layer.flops() / layer.min_memory())]

    df = pd.DataFrame(output,
                      columns=['Model', 'Impl', 'Dataset', 'Size',
                               'Flops', 'Gigaflops', 'Min memory bytes',
                               'Min memory gb', 'Op intensity'])
    df.to_csv(filename, index=False)


def compute_matmuls(layer_name, hidden_sizes, datasets, mm_ops, filename):
    # model, impl_name, dataset, hidden_size, val_dtype, idx_dtype
    output = []
    for dataset in datasets:
        for hidden_size in hidden_sizes:
            for mm_op in mm_ops:
                kwargs = {}
                if mm_op == measurable_ops.EllpackSpmm:
                    kwargs = {'max_row_size': dataset.max_row_size}
                layer = mm_op(N=dataset.num_nodes,
                              M=dataset.num_nodes,
                              F=hidden_size,
                              nnz=dataset.num_edges, val_dtype=dace.float32,
                              idx_dtype=dace.int32, **kwargs)
                print(f"{dataset.name}, {hidden_size}: {layer}")

                op_name = layer.__class__.__name__

                output += [
                    (layer_name, op_name, dataset.name, hidden_size,
                     layer.flops(),
                     layer.flops() / 1024 ** 3,
                     layer.min_memory(),
                     layer.min_memory() / 1024 ** 3,
                     layer.flops() / layer.min_memory())]

    df = pd.DataFrame(output,
                      columns=['Model', 'Op', 'Dataset', 'Size',
                               'Flops', 'Gigaflops', 'Min memory bytes',
                               'Min memory gb', 'Op intensity'])
    df.to_csv(filename, index=False)


def compare_fused_spmm_sddmm(hidden_sizes, datasets, filename):
    # model, impl_name, dataset, hidden_size, val_dtype, idx_dtype
    output = []
    for dataset in datasets:
        for hidden_size in hidden_sizes:
            for sddmm_class in [measurable_ops.CsrSddmm, measurable_ops.CooSddmm, measurable_ops.EllpackSddmm]:
                # fused = measurable_ops.FusedCooSpmmSddmm(N=dataset.num_nodes,
                #                                          M=hidden_size,
                #                                          nnz=dataset.num_edges, val_dtype=dace.float32,
                #                                          idx_dtype=dace.int32)
                kwargs = {}
                if sddmm_class == measurable_ops.EllpackSddmm:
                    kwargs = {'max_row_size': dataset.max_row_size}
                sddmm = sddmm_class(N=dataset.num_nodes, M=hidden_size,
                                    nnz=dataset.num_edges, val_dtype=dace.float32,
                                    idx_dtype=dace.int32, **kwargs)
                # spmm = measurable_ops.Coomm(N=dataset.num_nodes, M=dataset.num_nodes, F=hidden_size,
                #                               nnz=dataset.num_edges, val_dtype=dace.float32,
                #                               idx_dtype=dace.int32)

                print(f"{dataset.name}, {hidden_size}: {sddmm}")


def compare_semibatched_spmm_sddmm(sizes, datasets, outfile):
    batch_size = 8
    for hidden_size in sizes:
        for dataset in datasets:
            batched_spmm = measurable_ops.BatchedOp(
                measurable_ops.Coomm(N=dataset.num_nodes, M=dataset.num_nodes, F=hidden_size,
                                     nnz=dataset.num_edges, val_dtype=dace.float32,
                                     idx_dtype=dace.int32), batch_size)
            semibatched_spmm = measurable_ops.SemibatchedCoomm(B=batch_size, N=dataset.num_nodes, M=dataset.num_nodes,
                                                               F=hidden_size,
                                                               nnz=dataset.num_edges, val_dtype=dace.float32,
                                                               idx_dtype=dace.int32)
            print(f"{dataset.name}, {hidden_size}:")
            print(f"Batched: {batched_spmm}")
            print(f"Semibatched: {semibatched_spmm}")

            batched_sddmm = measurable_ops.BatchedOp(measurable_ops.CooSddmm(N=dataset.num_nodes, M=hidden_size,
                                                                             nnz=dataset.num_edges,
                                                                             val_dtype=dace.float32,
                                                                             idx_dtype=dace.int32), batch_size)
            semibatched_sddmm = measurable_ops.MultiheadCooSddmm(N=dataset.num_nodes, M=hidden_size,
                                                                 nnz=dataset.num_edges, val_dtype=dace.float32,
                                                                 idx_dtype=dace.int32, heads=batch_size)
            print(f"Batched: {batched_sddmm}")
            print(f"Semibatched: {semibatched_sddmm}")


def main():
    # Estimate the number of FLOPS and lower bound of memory movement for GAT
    # and GCN with typical sizes.
    datasets = [CoraStats, ArxivStats]

    compare_fused_spmm_sddmm([64, 128, 1024], datasets, '')

    matmuls = [measurable_ops.Csrmm, measurable_ops.Cscmm, measurable_ops.Coomm, measurable_ops.EllpackSpmm]

    compute_matmuls('basic', [64], datasets,
                    matmuls, 'basic-operators.csv')

    compare_semibatched_spmm_sddmm([64, 128, 1024], datasets, 'semibatched-operators.csv')

    hidden_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    gcn_layers = [GCNConvCSR, GCNConvCSRAdapt, GCNConvCOO, GCNConvCOOAdapt, GCNConvCSC,
                  GCNConvCSCAdapt]
    backward_gcn_layers = [BackwardGCNConvCSR, BackwardGCNConvCSRAdapt, BackwardGCNConvCOO,
                           BackwardGCNConvCOOAdapt, BackwardGCNConvCSC, BackwardGCNConvCSCAdapt]

    # compute_all('gcn_single_layer', hidden_sizes, datasets, gcn_layers,
    #             'gcn-numbers.csv', val_dtype=dace.float32,
    #             idx_dtype=dace.int32,
    #             do_bias=True)
    #
    # compute_all('gcn_single_layer', hidden_sizes, datasets,
    #             backward_gcn_layers, 'gcn-numbers-bwd.csv',
    #             val_dtype=dace.float32,
    #             idx_dtype=dace.int32,
    #             do_bias=True,
    #             compute_input_grad=True)
    #
    # gat_layers = [GATConvCSR, GATConvCOO]
    # compute_all('gat_single_layer', hidden_sizes, datasets, gat_layers,
    #             'gat-8_heads-numbers.csv', val_dtype=dace.float32,
    #             idx_dtype=dace.int32,
    #             heads=8,
    #             do_bias=True)
    #
    # compute_all('gat_single_layer', hidden_sizes, datasets, gat_layers,
    #             'gat-1_heads-numbers.csv', val_dtype=dace.float32,
    #             idx_dtype=dace.int32,
    #             heads=1,
    #             do_bias=True)


if __name__ == '__main__':
    main()
