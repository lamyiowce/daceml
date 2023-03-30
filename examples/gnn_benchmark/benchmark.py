import argparse
import dataclasses
import faulthandler
import functools
import logging
from pathlib import Path
from typing import Sequence, Dict, Optional, Type

import numpy as np
import torch
from torch_geometric.transforms import GCNNorm

import daceml
from daceml import onnx as donnx
from daceml.onnx import ONNXForward
from daceml.torch.module import dace_module
from examples.gnn_benchmark import sparse, models
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.datasets import get_dataset
from examples.gnn_benchmark.util import stats_as_csv_entry, create_dace_model, \
    register_replacement_overrides, make_torch_args, name_to_impl_class

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


def check_correctness(dace_models: Dict[str, ExperimentInfo],
                      torch_model: torch.nn.Module,
                      torch_edge_list_args,
                      torch_csr_args):
    torch_edge_list_pred = torch_model(*torch_edge_list_args)
    torch_csr_pred = torch_model(*torch_csr_args)
    torch_edge_list_pred = torch_edge_list_pred.detach().cpu()
    torch_csr_pred = torch_csr_pred.detach().cpu()
    if not torch.allclose(torch_edge_list_pred, torch_csr_pred, atol=1.0e-5):
        print("Torch edge list and torch csr results are not equal!")
        print("** Max abs error: ",
              abs((torch_edge_list_pred - torch_csr_pred)).max())
        print("** Avg abs error: ",
              abs((torch_edge_list_pred - torch_csr_pred)).mean())
        print("** Max rel error: ",
              (abs((torch_edge_list_pred - torch_csr_pred)) / abs(
                  torch_csr_pred)).max())

    all_correct = True
    for name, experiment_info in dace_models.items():
        print(f"---> Checking correctness for {name}...")
        model = experiment_info.model
        args = experiment_info.data.to_input_list()
        register_replacement_overrides(experiment_info.impl_name,
                                       experiment_info.gnn_type)
        if use_gpu:
            torch.cuda.nvtx.range_push(name + ' correctness')
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
                  (abs((dace_pred_cpu - torch_edge_list_pred)) / abs(
                      torch_edge_list_pred)).max())
            print(dace_pred_cpu)
            print(torch_edge_list_pred)
            experiment_info.correct = False
            all_correct = False

    correct_keys = [key for key, value in dace_models.items() if value.correct]
    incorrect_keys = [key for key, value in dace_models.items() if
                      not value.correct]

    print(f"\n☆ =================== SUMMARY ================== ☆")
    if len(correct_keys) > 0:
        print(f"☆ ============================================== ☆")
        print(
            f"==== Results correct for {', '.join(correct_keys)}. ☆ ╰(o＾◡＾o)╯ ☆ ====")
        print(f"☆ ============================================== ☆")
    if len(incorrect_keys) > 0:
        print(
            f"****** INCORRECT RESULTS FOR {', '.join(incorrect_keys)}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print(f"****************************************************\n")

    return all_correct


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 save_output: bool = True,
                 small: bool = False):
    from examples.gnn_benchmark.performance_measurement import \
        print_time_statistics
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

    print(f"---> Benchmarking...")
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
                stats_as_csv_entry(times, func_names, args.model,
                                   args.hidden))




def main():
    model_dict = {'gcn': models.GCN, 'linear': models.LinearModel,
                  'gat': models.GAT, 'gcn_single_layer': models.GCNSingleLayer}

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'dry', 'onlydace',
                                           'benchmark_small'],
                        required=True)
    parser.add_argument('--impl', type=str, nargs='+', required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--persistent-mem', action='store_true')
    parser.add_argument('--opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--skip-check', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys())
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    args = parser.parse_args()

    model_class = model_dict[args.model]
    num_hidden_features = args.hidden
    args.outfile = Path(args.outfile) if args.outfile is not None else None

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    data = get_dataset(args.data, device)

    print("Num node features: ", data.num_node_features)
    num_classes = data.y.max().item() + 1
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    print("Num nodes:", data.num_nodes)
    print("Num non zero:", data.num_edges)
    normalize = args.normalize
    print("Normalize: ", normalize)
    print("Implementation: ", args.impl)

    # Define models.
    model_args = (data.num_node_features, num_hidden_features, num_classes,
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
                                       gnn_implementation_name=impl_name,
                                       threadblock_dynamic=args.threadblock_dynamic,
                                       persistent_mem=args.persistent_mem,
                                       do_opt=args.opt,
                                       device=device)
        info = ExperimentInfo(impl_name=impl_name,
                              implementation=implementation_class,
                              model=dace_model,
                              data_format=implementation_class.graph_format,
                              gnn_type=args.model)
        dace_models[impl_name] = info

    if 'gcn' in args.model:
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    torch_model, dace_models = optimize_data(torch_model, dace_models, data)
    for k, v in dace_models.items():
        print(f"Impl: {k}")
        print(v)

    if args.mode == 'only_dace':
        print('Running only dace models.')
        for dace_model_name, dace_model_info in dace_models.items():
            model = dace_model_info.model
            inputs = dace_model_info.data.to_input_list()
            result = model(*inputs)
            if use_gpu:
                torch.cuda.synchronize()
            print(f"Dace {dace_model_name}: ", result)
    else:
        torch_csr_args, torch_edge_list_args = make_torch_args(data,
                                                               model_name=args.model)

        results_correct = check_correctness(dace_models, torch_model,
                                            torch_edge_list_args,
                                            torch_csr_args)

        if args.mode == 'benchmark' or args.mode == 'benchmark_small':
            print("Benchmarking...")
            do_benchmark(dace_models, torch_model, torch_csr_args,
                         torch_edge_list_args, args,
                         save_output=results_correct,
                         small=args.mode == 'benchmark_small')


if __name__ == '__main__':
    main()
