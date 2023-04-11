import argparse
import copy
import dataclasses
import faulthandler
import functools
import logging
import pathlib
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
    correct_grads: Optional[bool] = None
    data: Optional[sparse.GraphMatrix] = None


def check_equal(result, expected, name_result=None, name_expected=None,
                verbose=True):
    name_result = name_result or 'result'
    name_expected = name_expected or 'expected'
    if torch.allclose(expected, result, atol=1.0e-5):
        if verbose:
            print(
                f"==== Correct: {name_result}.  ☆ ╰(o＾◡＾o)╯ ☆ ====")
    else:
        print(
            f"****** INCORRECT: {name_result}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print("** Max abs error: ",
              abs((expected - result)).max().item())
        print("** Avg abs error: ",
              abs((expected - result)).mean().item())
        print("** Max rel error: ",
              (abs((expected - result)) / abs(
                  result)).max().item())
        print("** Avg rel error: ", (abs((expected - result)) / abs(
            result)).mean().item())
        print(f"** {name_result}:", result)
        print(f"** {name_expected}:", expected)
        return False

    return True


def check_gradients(result_model: torch.nn.Module,
                    expected_model: torch.nn.Module,
                    name_result: str,
                    name_expected: str,
                    verbose=True) -> bool:
    result_parameters = dict(result_model.named_parameters())
    all_correct = True
    for name, parameter in expected_model.named_parameters():
        result_grad = result_parameters[name].grad
        all_correct &= check_equal(result_grad, parameter.grad,
                                   name_expected=name_expected + ": " + name,
                                   name_result=name_result + ": " + name,
                                   verbose=verbose)
    return all_correct


def check_correctness(dace_models: Dict[str, ExperimentInfo],
                      torch_model: torch.nn.Module,
                      torch_edge_list_args,
                      torch_csr_args,
                      targets: torch.Tensor,
                      backward: bool) -> bool:
    torch_model_csr = copy.deepcopy(torch_model)
    torch_model.train()
    torch_model_csr.train()

    torch_edge_list_pred = torch_model(*torch_edge_list_args)
    torch_csr_pred = torch_model_csr(*torch_csr_args)
    check_equal(torch_csr_pred.detach(), torch_edge_list_pred.detach(),
                verbose=False)

    def backward_func(pred):
        loss = criterion(pred, targets)
        loss.backward()

    if backward:
        if hasattr(torch_model, 'conv2'):
            criterion = torch.nn.NLLLoss()
        else:
            criterion = lambda pred, targets: torch.sum(pred)

        backward_func(torch_edge_list_pred)
        backward_func(torch_csr_pred)

        check_gradients(torch_model_csr, torch_model, "CSR", "EdgeList",
                        verbose=False)

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

        experiment_info.correct = check_equal(dace_pred,
                                              torch_edge_list_pred,
                                              name_result=f"Predictions for DaCe {name}",
                                              name_expected="Torch predictions")
        if backward:
            backward_func(dace_pred)
            experiment_info.correct_grads = check_gradients(model.model,
                                                            torch_model,
                                                            name_result=f"Gradients for DaCe {name}",
                                                            name_expected="Torch gradients")

    correct_keys = [key for key, value in dace_models.items() if value.correct]
    incorrect_keys = [key for key, value in dace_models.items() if
                      not value.correct]

    print(f"\n☆ =================== SUMMARY ================== ☆")
    if len(correct_keys) > 0:
        print(f"==== Predictions correct for {', '.join(correct_keys)}"
              f". ☆ ╰(o＾◡＾o)╯ ☆ ====")
        print(f"☆ ============================================== ☆")
    if len(incorrect_keys) > 0:
        print(f"****** INCORRECT PREDICTIONS FOR {', '.join(incorrect_keys)}"
              f"! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print(f"****************************************************\n")

    if backward:
        grads_correct_keys = [key for key, value in dace_models.items() if
                              value.correct_grads]
        grads_incorrect_keys = [key for key, value in dace_models.items() if
                                not value.correct_grads]
        if len(grads_correct_keys) > 0:
            print(
                f"==== Gradients correct for {', '.join(grads_correct_keys)}"
                f". ☆ ╰(o＾◡＾o)╯ ☆ ====")
            print(f"☆ ============================================== ☆")
        if len(grads_incorrect_keys) > 0:
            print(
                f"****** INCORRECT GRADIENTS FOR {', '.join(grads_incorrect_keys)}"
                f"! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
            print(f"****************************************************\n")
        return len(incorrect_keys) == 0 and len(grads_incorrect_keys) == 0

    return len(incorrect_keys) == 0


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 targets: Optional[torch.Tensor],
                 backward: bool,
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
        return model(*inputs)

    for experiment_info in experiment_infos.values():
        model = experiment_info.model
        # TODO: check eval and train modes
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

    print(f"---> Benchmarking the forward pass...")
    times = measure_performance(funcs,
                                func_names=func_names,
                                warmups=10 if not small else 2,
                                num_iters=10 if not small else 2,
                                timing_iters=100 if not small else 3)
    print()
    print(f"\n------ {args.model.upper()} FORWARD RUNTIME [ms] ------")
    print_time_statistics(times, func_names)
    print()

    if args.outfile is not None and save_output:
        write_stats_to_file(args, func_names, times, file_path=args.outfile)

    if backward:
        assert targets is not None

        if hasattr(torch_model, 'conv2'):
            criterion = torch.nn.NLLLoss()
        else:
            criterion = lambda pred, targets: torch.sum(pred)

        def backward_fn(pred_fn):
            pred = pred_fn()
            loss = criterion(pred, targets)
            loss.backward()

        backward_funcs = [functools.partial(backward_fn, f) for f in funcs]

        print(f"---> Benchmarking the BACKWARD pass...")
        times = measure_performance(backward_funcs,
                                    func_names=func_names,
                                    warmups=10 if not small else 2,
                                    num_iters=10 if not small else 2,
                                    timing_iters=100 if not small else 3)
        print()
        print(f"\n------ {args.model.upper()} BACKWARD RUNTIME [ms] ------")
        print_time_statistics(times, func_names)
        print()

        if args.outfile and save_output:
            path = args.outfile.with_name(args.outfile.stem + "-bwd.csv")
            write_stats_to_file(args, func_names, times, path)


def write_stats_to_file(args, func_names, times, file_path: pathlib.Path):
    add_header = not file_path.exists()
    with open(file_path, 'a') as file:
        if add_header:
            headers = [
                'Name', 'Model', 'Size', 'Min', 'Mean',
                'Median',
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
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys())
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    parser.add_argument('--no-gen-code', action='store_true')
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
    torch_model = model_class(*model_args, bias_init=torch.nn.init.uniform_).to(device)
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
                                       device=device,
                                       gen_code=not args.no_gen_code,
                                       backward=args.backward)
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
                                            torch_csr_args,
                                            targets=data.y,
                                            backward=args.backward)

        if args.mode == 'benchmark' or args.mode == 'benchmark_small':
            do_benchmark(dace_models, torch_model, torch_csr_args,
                         torch_edge_list_args, args,
                         backward=args.backward,
                         targets=data.y,
                         save_output=results_correct,
                         small=args.mode == 'benchmark_small')


if __name__ == '__main__':
    main()
