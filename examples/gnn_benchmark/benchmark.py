import argparse
import copy
import dataclasses
import faulthandler
import functools
import logging
import statistics
from pathlib import Path
from typing import Sequence, Dict, Optional, Type

import dace
import numpy as np
import torch
from torch_geometric.transforms import GCNNorm
from torch_sparse import SparseTensor

import daceml
from daceml import onnx as donnx
from daceml.onnx import register_replacement, ONNXForward
from daceml.onnx.nodes import replacement_entries
from daceml.torch.module import dace_module, DaceModule
from examples.gnn_benchmark import sdfg_util, sparse, models, datasets
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.datasets import get_dataset
from examples.gnn_benchmark.implementations import gat_implementations
from examples.gnn_benchmark.implementations import gcn_implementations
from examples.gnn_benchmark.implementations.common import SparseLayerBase
from examples.gnn_benchmark.sdfg_util import specialize_mem_onnx, \
    apply_dace_auto_optimize, make_maps_dynamic

faulthandler.enable()
donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


@dataclasses.dataclass
class ExperimentInfo:
    impl_name: str
    gnn_type: str
    implementation: ONNXForward
    data_format: Type[sparse.GraphMatrix]
    model: daceml.torch.DaceModule
    correct: Optional[bool] = None
    data: Optional[sparse.GraphMatrix] = None


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


def check_correctness(dace_models: Dict[str, ExperimentInfo],
                      torch_model: torch.nn.Module,
                      torch_edge_list_args,
                      torch_csr_args):
    torch_edge_list_pred = torch_model(*torch_edge_list_args)
    torch_csr_pred = torch_model(*torch_csr_args)
    torch_edge_list_pred = torch_edge_list_pred.detach().cpu()
    torch_csr_pred = torch_csr_pred.detach().cpu()
    assert torch.allclose(torch_edge_list_pred, torch_csr_pred, atol=1.0e-5)

    all_correct = True
    for name, experiment_info in dace_models.items():
        print(f"---> Checking correctness for {name}...")
        model = experiment_info.model
        args = experiment_info.data.to_input_list()
        register_replacement_overrides(experiment_info.impl_name,
                                       experiment_info.gnn_type)
        if use_gpu:
            torch.cuda.nvtx.range_push(name + ' Correctness')
        dace_pred = model(*args)
        if use_gpu:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        dace_pred_cpu = dace_pred.detach().cpu()
        if np.allclose(dace_pred_cpu, torch_edge_list_pred, atol=1.0e-5):
            print(f"\n==== Results correct for {name}.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
            experiment_info.correct = True
        else:
            print(f"\n****** INCORRECT RESULTS FOR {name}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
            print("** Max abs error: ",
                  abs((dace_pred_cpu - torch_edge_list_pred)).max())
            print("** Avg abs error: ",
                  abs((dace_pred_cpu - torch_edge_list_pred)).mean())
            print("** Max rel error: ",
                  (abs((dace_pred_cpu - torch_edge_list_pred)) / abs(torch_edge_list_pred)).max())
            print(dace_pred_cpu)
            print(torch_edge_list_pred)
            experiment_info.correct = False
            all_correct = False

    correct_keys = [key for key, value in dace_models.items() if value.correct]
    incorrect_keys = [key for key, value in dace_models.items() if not value.correct]

    print(f"\n☆ =================== SUMMARY ================== ☆")
    if len(correct_keys) > 0:
        print(f"☆ ============================================== ☆")
        print(f"==== Results correct for {', '.join(correct_keys)}. ☆ ╰(o＾◡＾o)╯ ☆ ====")
        print(f"☆ ============================================== ☆")
    if len(incorrect_keys) > 0:
        print(f"****** INCORRECT RESULTS FOR {', '.join(incorrect_keys)}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print(f"****************************************************\n")

    return all_correct


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 save_output: bool = True,
                 small: bool = False):
    from examples.gnn_benchmark.performance_measurement import print_time_statistics
    if use_gpu:
        from examples.gnn_benchmark.performance_measurement import \
        measure_performance
    else:
        from examples.gnn_benchmark.performance_measurement import \
            measure_performance_cpu as measure_performance

    funcs = [
        lambda: torch_model(*torch_csr_args),
        lambda: torch_model(*torch_edge_list_args),
    ]
    func_names = ['torch_csr', 'torch_edge_list']

    def run_with_inputs(model, inputs):
        model(*inputs)

    for experiment_info in experiment_infos.values():
        model = experiment_info.model
        inputs = experiment_info.data.to_input_list()

        funcs.append(functools.partial(run_with_inputs, model, inputs))
        name = "dace"
        if args.threadblock_dynamic:
            name += "_tb-dynamic"
        if args.opt:
            name += "_autoopt"
        if args.persistent_mem:
            name += "_persistent_mem"
        name += f"_{experiment_info.impl_name}"
        func_names.append(name)

    times = measure_performance(funcs,
                                func_names=func_names,
                                warmups=10 if not small else 2,
                                num_iters=10 if not small else 2,
                                timing_iters=100 if not small else 3)
    print()
    print(f"\n------ {args.model.upper()} RUNTIME [ms] ------")
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


def create_dace_model(model: torch.nn.Module,
                      state_dict: Dict[str, torch.Tensor],
                      gnn_implementation_name: str,
                      do_opt: bool,
                      persistent_mem: bool,
                      threadblock_dynamic: bool) -> dace.DaceModule:
    sdfg_name = f"{model.__class__.__name__}_{gnn_implementation_name}"
    dace_model = DaceModule(copy.deepcopy(model), sdfg_name=sdfg_name).to(device)

    dace_model.model.load_state_dict(state_dict)

    dace_model.eval()

    if do_opt:
        print("---> Adding auto-opt hook.")
        dace_model.append_post_onnx_hook("dace_auto_optimize",
                                         apply_dace_auto_optimize)

    if persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)

    if threadblock_dynamic:
        print("---> Adding threadblock dynamic maps hook.")
        exclude_loops = []
        if isinstance(model, models.GCN):
            # Has to be skipped, otherwise the computation results are incorrect.
            exclude_loops = [
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_45_4_46'
            ]
        elif isinstance(model, models.GAT):
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

    set_implementation = functools.partial(
        sdfg_util.set_implementation, implementation_name=gnn_implementation_name)
    dace_model.prepend_post_onnx_hook("set_implementation",
                                      set_implementation)

    return dace_model


name_to_impl_class: Dict[str, Dict[str, SparseLayerBase]] = {
    "gcn": {"csr": gcn_implementations.GCNConvCSR,
            "coo": gcn_implementations.GCNConvCOO,
            "csc": gcn_implementations.GCNConvCSC,
            "ellpack_t": gcn_implementations.GCNConvEllpackTransposed,
            "ellpack": gcn_implementations.GCNConvEllpack,
            "semester_thesis": gcn_implementations.GCNConvSemesterThesis},
    "gat": {"csr": gat_implementations.GATConvCSR,
            "semester_thesis": gat_implementations.GATConvSemesterThesis}
}
name_to_impl_class['gcn_single_layer'] = name_to_impl_class['gcn']


def register_replacement_overrides(implementation_name, layer_name):
    impl_class = name_to_impl_class[layer_name][implementation_name]
    input_spec = impl_class.input_spec
    if 'gcn' in layer_name:
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
    model_dict = {'gcn': models.GCN, 'linear': models.LinearModel,
                  'gat': models.GAT, 'gcn_single_layer': models.GCNSingleLayer}

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', choices=list(datasets.dataset_classes.keys()) + ['small'], default='cora')
    parser.add_argument('--mode', choices=['benchmark', 'dry', 'onlydace', 'benchmark_small'],
                        required=True)
    parser.add_argument('--impl', type=str, nargs='+', required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys())
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    args = parser.parse_args()

    model_class = model_dict[args.model]
    num_hidden_features = args.hidden
    args.hidden = args.hidden or (8 if args.model == 'gat' else 512)
    args.outfile = Path(args.outfile) if args.outfile is not None else None

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    data, num_node_features, num_classes = get_dataset(args.data, device)

    print("Num node features: ", num_node_features)
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    normalize = args.normalize
    print("Normalize: ", normalize)
    print("Implementation: ", args.impl)

    # Define models.
    model_args = (num_node_features, num_hidden_features, num_classes,
                  normalize)
    torch_model = model_class(*model_args).to(device)
    torch_model.eval()

    dace_models = {}
    available_implementations = name_to_impl_class[args.model]
    if args.impl == ['none']:
        available_implementations = {}
    elif args.impl != ['all']:
        available_implementations = {
            impl: available_implementations[impl]
            for impl in args.impl
        }

    for impl_name, implementation_class in available_implementations.items():
        dace_model = create_dace_model(torch_model,
                                       state_dict=torch_model.state_dict(),
                                       gnn_implementation_name=impl_name,
                                       threadblock_dynamic=args.threadblock_dynamic,
                                       persistent_mem=args.persistent_mem,
                                       do_opt=args.opt)
        dace_models[impl_name] = ExperimentInfo(impl_name=impl_name,
                                                implementation=implementation_class,
                                                model=dace_model,
                                                data_format=implementation_class.graph_format,
                                                gnn_type=args.model)

    if 'gcn' in args.model:
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    x = data.x
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight)
    edge_rowptr, edge_col, edge_weights = sparse_edge_index.csr()
    print("Num non zero:", edge_col.shape[0])

    torch_model, dace_models = optimize_data(torch_model, dace_models, data)
    print(dace_models)

    # Create args lists for torch models.
    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = x, sparse_edge_index.t()

    torch_edge_list_args = x, data.edge_index
    if edge_weights is not None and 'gcn' in args.model:
        torch_edge_list_args += (data.edge_weight,)

    results_correct = check_correctness(dace_models, torch_model,
                                        torch_edge_list_args, torch_csr_args)

    if args.mode == 'onlydace':
        print('Only dace models for profiling.')
        for dace_model_name, dace_model_info in dace_models.items():
            model = dace_model_info.dace_model
            inputs = dace_model_info.data.to_input_list()
            result = model(*inputs)
            if use_gpu:
                torch.cuda.synchronize()
            print(f"Dace {dace_model_name}: ", result)
    elif args.mode == 'dry':
        print("Single run of all models.")
        for dace_model_name, dace_model_info in dace_models.items():
            model = dace_model_info.model
            inputs = dace_model_info.data.to_input_list()
            result = model(*inputs)
            if use_gpu:
                torch.cuda.synchronize()
            print(f"Dace {dace_model_name}: ", result)
        print("PyG csr: ", torch_model(*torch_csr_args))
        print("PyG edge list: ", torch_model(*torch_edge_list_args))
    elif args.mode == 'benchmark' or args.mode == 'benchmark_small':
        print("Benchmarking...")
        do_benchmark(dace_models, torch_model, torch_csr_args,
                     torch_edge_list_args, args, save_output=results_correct, small=args.mode == 'benchmark_small')
    else:
        raise ValueError("Mode not supported " + args.mode)


if __name__ == '__main__':
    main()
