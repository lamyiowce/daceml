import argparse
import faulthandler
import functools
import logging
import statistics
from pathlib import Path
from typing import Sequence, Tuple

import dace
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import GCNNorm
from torch_sparse import SparseTensor

import daceml
from daceml import onnx as donnx
from daceml.onnx import register_replacement
from daceml.onnx.nodes import replacement_entries
from daceml.onnx.op_implementations import replacement_implementations
from daceml.torch.module import dace_module
from examples.gnn_benchmark import util
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.models import LinearModel, GCN, GAT
from examples.gnn_benchmark.util import specialize_mem_onnx, \
    apply_dace_auto_optimize, make_maps_dynamic

faulthandler.enable()
donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stats_as_string(times, func_names, model, hidden_size):
    """ Print timing statistics.
    :param times: the result of time_funcs.
    :param func_names: a name to use for each function timed.
    """

    out = ''
    for name, func_time, in zip(func_names, times):
        row = [
            name, model, hidden_size,
            min(func_time),
            statistics.mean(func_time),
            statistics.median(func_time),
            statistics.stdev(func_time) if len(func_time) > 1 else 0.0,
            max(func_time)
        ]
        out += ','.join(map(str, row)) + '\n'
    return out


def check_correctness(dace_model, torch_model, dace_args,
                      torch_edge_list_args):
    dace_pred = dace_model(*dace_args)
    torch_pred = torch_model(*torch_edge_list_args)
    dace_pred_cpu = dace_pred.detach().cpu()
    torch_pred_cpu = torch_pred.detach().cpu()
    if np.allclose(dace_pred_cpu, torch_pred_cpu, atol=1.0e-5):
        print("\n==== Results correct.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
        return True
    else:
        print("\n****** INCORRECT RESULTS! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print("Max abs error: ", abs((dace_pred_cpu - torch_pred_cpu)).max())
        print(dace_pred_cpu - torch_pred_cpu)
        return False


def do_benchmark(dace_model: daceml.torch.DaceModule,
                 dace_args: Sequence[torch.Tensor],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 save_output: bool = True):
    from daceml.testing.profiling import time_funcs, print_time_statistics

    funcs = [
        lambda: dace_model(*dace_args),
        lambda: torch_model(*torch_csr_args),
        lambda: torch_model(*torch_edge_list_args),
    ]

    name = args.name
    if args.threadblock_dynamic:
        name += "_tb-dynamic"
    if args.opt:
        name += "_autoopt"
    if args.persistent_mem:
        name += "_persistent_mem"
    func_names = [name, 'torch_csr', 'torch_edge_list']
    times = time_funcs(funcs,
                       func_names=func_names,
                       warmups=10,
                       num_iters=100)
    print()
    print(f"\n------ {args.model.upper()} ------")
    print_time_statistics(times, func_names)
    print()

    if args.outfile is not None and save_output:
        add_header = not args.outfile.exists()
        with open(args.outfile, 'a') as file:
            if add_header:
                headers = [
                    'Name', 'Model', 'Size', 'Min', 'Mean', 'Median',
                    'Stdev', 'Max'
                ]
                file.write(','.join(headers) + '\n')
            file.write(
                stats_as_string(times, func_names, args.model,
                                args.hidden))


def get_dataset(dataset_name: str) -> Tuple[
    torch_geometric.data.Data, int, int]:
    if dataset_name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0].to(device)
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
    elif dataset_name == 'small':
        _x = torch.tensor([[0., 1], [1, 1], [-1, 0]]).to(device)
        _edge_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0,
                                                      2]]).to(device)
        _edge_attr = torch.tensor([1, 1, 1, 1., 1]).to(device)
        data = Data(x=_x, edge_index=_edge_index, edge_attr=_edge_attr)
        num_node_features = _x.shape[1]
        num_classes = 2
    else:
        raise NotImplementedError("No such dataset: ", dataset_name)
    return data, num_node_features, num_classes


def register_replacement_overrides(implementation_name, layer_name):
    name_to_impl_class = {
        "gcn": {"csr": replacement_implementations.GCNConvCSR,
                "coo": replacement_implementations.GCNConvCOO},
        "gat": {"csr": replacement_implementations.GATConvCSR,
                "semester_thesis": replacement_implementations.GATConvSemesterThesis}
    }
    input_spec = name_to_impl_class[layer_name][
        implementation_name].get_input_spec()
    if layer_name == 'gcn':
        register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                             inputs=input_spec,
                             outputs={'output': dace.float32},
                             shape_infer=replacement_entries.shape_infer_GCNConv,
                             shape_fn_from_module=replacement_entries.make_GCNConv_shape_fn)
    elif layer_name == 'gat':
        register_replacement('torch_geometric.nn.conv.gat_conv.GATConv',
                             inputs=input_spec,
                             outputs={'output': dace.float32},
                             shape_infer=replacement_entries.shape_infer_GATConv,
                             shape_fn_from_module=replacement_entries.make_GATConv_shape_fn)


def main():
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', choices=['small', 'cora'], default='cora')
    parser.add_argument('--mode', choices=['benchmark', 'dry', 'onlydace'],
                        required=True)
    parser.add_argument('--impl', type=str, required=True)
    parser.add_argument('--target_format', choices=['csr', 'coo'],
                        required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--model', choices=['gcn', 'gat', 'linear'])
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    args = parser.parse_args()
    models = {'gcn': GCN, 'linear': LinearModel, 'gat': GAT}
    model_class = models[args.model]
    num_hidden_features = args.hidden
    args.hidden = args.hidden or (8 if args.model == 'gat' else 512)
    args.outfile = Path(args.outfile) if args.outfile is not None else None

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    data, num_node_features, num_classes = get_dataset(args.data)

    print("Num node features: ", num_node_features)
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    normalize = args.normalize
    print("Normalize: ", normalize)
    print("Target data format: ", args.target_format)

    register_replacement_overrides(implementation_name=args.impl,
                                   layer_name=args.model)

    # Define models.
    torch_model = model_class(num_node_features, num_hidden_features,
                              num_classes, normalize).to(device)
    dace_model = dace_module(model_class)(num_node_features,
                                          num_hidden_features, num_classes,
                                          normalize).to(device)

    dace_model.model.load_state_dict(torch_model.state_dict())

    dace_model.eval()
    torch_model.eval()

    if args.opt:
        print("---> Adding auto-opt hook.")
        dace_model.append_post_onnx_hook("dace_auto_optimize",
                                         apply_dace_auto_optimize)

    if args.persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)

    if args.threadblock_dynamic:
        print("---> Adding threadblock dynamic maps hook.")
        exclude_loops = []
        if args.model == 'gcn':
            # Has to be skipped, otherwise the computation results are incorrect.
            exclude_loops = [
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_45_4_46'
            ]
        elif args.model == 'gat':
            exclude_loops = [
                # Below two have to be excluded because only one-dimensional
                # maps are supported in DaCe for dynamic block map schedule
                # (got 2).
                '_Div__map',
                '_Mult__map',
                # Below two have to be excluded, otherwise compile errors
                # occur (the generated code is incorrect).
                'assign_137_12_map',
                'outer_fused',
            ]
        make_maps_dynamic_with_excluded_loops = functools.partial(
            make_maps_dynamic, exclude_loops=exclude_loops)
        dace_model.append_post_onnx_hook(
            "apply_threadblock_dynamic_maps",
            make_maps_dynamic_with_excluded_loops)

    set_implementation = functools.partial(util.set_implementation,
                                           implementation_name=args.impl)
    dace_model.prepend_post_onnx_hook("set_implementation", set_implementation)

    if args.model == 'gcn':
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)
    x = data.x
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight)
    edge_rowptr, edge_col, edge_weights = sparse_edge_index.csr()

    torch_model, dace_data = optimize_data(torch_model, data,
                                           target_format=args.target_format)
    print(dace_data)
    dace_args = (x,) if args.model == 'linear' else dace_data.to_input_list()

    # Create args lists for torch models.
    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = (x,) if args.model == 'linear' else (
        x, sparse_edge_index.t())

    torch_edge_list_args = (x,) if args.model == 'linear' else (
        x, data.edge_index)
    if edge_weights is not None and args.model == 'gcn':
        torch_edge_list_args += (data.edge_weight,)

    results_correct = check_correctness(dace_model, torch_model, dace_args,
                                        torch_edge_list_args)

    if args.mode == 'onlydace':
        print('Only dace model for profiling.')
        print("Dace: ", dace_model(*dace_args))
    elif args.mode == 'dry':
        print("Single run of all models.")
        print("Dace: ", dace_model(*dace_args))
        print("PyG csr: ", torch_model(*torch_csr_args))
        print("PyG edge list: ", torch_model(*torch_edge_list_args))
    elif args.mode == 'benchmark':
        print("Benchmarking...")
        do_benchmark(dace_model, dace_args, torch_model, torch_csr_args,
                     torch_edge_list_args, args, save_output=results_correct)
    else:
        raise ValueError("Mode not supported " + args.mode)


if __name__ == '__main__':
    main()
