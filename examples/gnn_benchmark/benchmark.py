import argparse
import dataclasses
import faulthandler
import functools
import logging
import statistics
from pathlib import Path
from typing import Sequence, Tuple, Dict, Optional, Type

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
from daceml.torch.module import dace_module, DaceModule
from examples.gnn_benchmark import util, sparse, models
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.util import specialize_mem_onnx, \
    apply_dace_auto_optimize, make_maps_dynamic

faulthandler.enable()
donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class ExperimentInfo:
    impl_name: str
    implementation: replacement_implementations.ONNXForward
    data_format: str
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


def check_correctness(dace_models: Dict[str, dace.DaceModule],
                      torch_model: torch.nn.Module,
                      torch_edge_list_args):
    torch_pred = torch_model(*torch_edge_list_args)
    torch_pred_cpu = torch_pred.detach().cpu()
    all_correct = True
    for name, experiment_info in dace_models.items():
        model = experiment_info.model
        args = experiment_info.data.to_input_list()
        register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                             inputs=experiment_info.implementation.get_input_spec(),
                             outputs={'output': dace.float32},
                             shape_infer=replacement_entries.shape_infer_GCNConv,
                             shape_fn_from_module=replacement_entries.make_GCNConv_shape_fn)
        dace_pred = model(*args)
        dace_pred_cpu = dace_pred.detach().cpu()
        if np.allclose(dace_pred_cpu, torch_pred_cpu, atol=1.0e-5):
            print(f"\n==== Results correct for {name}.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
            experiment_info.correct = True
        else:
            print(f"\n****** INCORRECT RESULTS FOR {name}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
            print("Max abs error: ",
                  abs((dace_pred_cpu - torch_pred_cpu)).max())
            print(dace_pred_cpu - torch_pred_cpu)
            experiment_info.correct = False
            all_correct = False

    if all_correct and len(dace_models) > 1:
        print(f"\n☆ ============================================== ☆")
        print(
            f"==== Results correct for {', '.join(dace_models.keys())}. ☆ ╰(o＾◡＾o)╯ ☆ ====")
        print(f"☆ ============================================== ☆")

    return all_correct


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 save_output: bool = True):
    from daceml.testing.profiling import print_time_statistics
    from examples.gnn_benchmark.performance_measurement import \
        measure_performance

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
                                warmups=50,
                                num_iters=500)
    print()
    print(f"\n------ fixed timing {args.model.upper()} ------")
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


def create_dace_model(model_class: Type[torch.nn.Module],
                      num_node_features: int, num_hidden_features: int,
                      num_classes: int, normalize: bool,
                      state_dict: Dict[str, torch.Tensor],
                      gnn_implementation_name: str,
                      do_opt: bool,
                      persistent_mem: bool,
                      threadblock_dynamic: bool) -> dace.DaceModule:
    sdfg_name = f"{model_class.__name__}_{gnn_implementation_name}"
    dace_model = DaceModule(model_class(
        num_node_features,
        num_hidden_features, num_classes,
        normalize), sdfg_name=sdfg_name).to(device)

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
        if model_class == models.GCN:
            # Has to be skipped, otherwise the computation results are incorrect.
            exclude_loops = [
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_45_4_46'
            ]
        elif model_class == models.GAT:
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
        util.set_implementation, implementation_name=gnn_implementation_name)
    dace_model.prepend_post_onnx_hook("set_implementation",
                                      set_implementation)

    return dace_model


name_to_impl_class = {
    "gcn": {"csr": (replacement_implementations.GCNConvCSR, sparse.CsrGraph),
            "coo": (replacement_implementations.GCNConvCOO, sparse.CooGraph),
            "csc": (replacement_implementations.GCNConvCSC, sparse.CscGraph),
            "ellpack_t": (replacement_implementations.GCNConvEllpackTransposed,
                          sparse.EllpackTransposedGraph),
            "ellpack": (replacement_implementations.GCNConvEllpack,
                        sparse.EllpackGraph)},
    "gat": {"csr": (replacement_implementations.GATConvCSR, sparse.CsrGraph),
            "semester_thesis": (
                replacement_implementations.GATConvSemesterThesis,
                sparse.CsrGraph)}
}


def register_replacement_overrides(implementation_name, layer_name):
    impl_class, _ = name_to_impl_class[layer_name][implementation_name]
    input_spec = impl_class.get_input_spec()
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
    parser.add_argument('--impl', type=str, nargs='+', required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--model', choices=['gcn', 'gat', 'linear'])
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    args = parser.parse_args()
    model_dict = {'gcn': models.GCN, 'linear': models.LinearModel,
                  'gat': models.GAT}
    model_class = model_dict[args.model]
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
    print("Implementation: ", args.impl)

    # Define models.
    torch_model = model_class(num_node_features, num_hidden_features,
                              num_classes, normalize).to(device)
    torch_model.eval()

    dace_models = {}
    available_implementations = name_to_impl_class[args.model]
    if args.impl != ['all']:
        available_implementations = {
            impl: available_implementations[impl]
            for impl in args.impl
        }

    for impl_name, (
            implementation_class,
            data_format) in available_implementations.items():
        dace_model = create_dace_model(model_class, num_node_features,
                                       num_hidden_features, num_classes,
                                       normalize=normalize,
                                       state_dict=torch_model.state_dict(),
                                       gnn_implementation_name=impl_name,
                                       threadblock_dynamic=args.threadblock_dynamic,
                                       persistent_mem=args.persistent_mem,
                                       do_opt=args.opt)
        dace_models[impl_name] = ExperimentInfo(impl_name=impl_name,
                                                implementation=implementation_class,
                                                model=dace_model,
                                                data_format=data_format)

    if args.model == 'gcn':
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    x = data.x
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight)
    edge_rowptr, edge_col, edge_weights = sparse_edge_index.csr()

    torch_model, dace_models = optimize_data(torch_model, dace_models, data)
    print(dace_models)

    # Create args lists for torch models.
    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = x, sparse_edge_index.t()

    torch_edge_list_args = x, data.edge_index
    if edge_weights is not None and args.model == 'gcn':
        torch_edge_list_args += (data.edge_weight,)

    results_correct = check_correctness(dace_models, torch_model,
                                        torch_edge_list_args)

    if args.mode == 'onlydace':
        print('Only dace models for profiling.')
        for dace_model_name, dace_model_info in dace_models.items():
            model = dace_model_info.dace_model
            inputs = dace_model_info.data.to_input_list()
            print(f"Dace {dace_model_name}: ", model(*inputs))
    elif args.mode == 'dry':
        print("Single run of all models.")
        for dace_model_name, dace_model_info in dace_models.items():
            model = dace_model_info.model
            inputs = dace_model_info.data.to_input_list()
            print(f"Dace {dace_model_name}: ", model(*inputs))
        print("PyG csr: ", torch_model(*torch_csr_args))
        print("PyG edge list: ", torch_model(*torch_edge_list_args))
    elif args.mode == 'benchmark':
        print("Benchmarking...")
        do_benchmark(dace_models, torch_model, torch_csr_args,
                     torch_edge_list_args, args, save_output=results_correct)
    else:
        raise ValueError("Mode not supported " + args.mode)


if __name__ == '__main__':
    main()
