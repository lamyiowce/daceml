import argparse
import dataclasses
import faulthandler
import functools
import logging
import pathlib
import socket
import subprocess
from pathlib import Path
from typing import Sequence, Dict, Optional, Callable
from collections import OrderedDict

import numpy as np
import torch
import torch_geometric
from torch_geometric.transforms import GCNNorm

import daceml
from daceml import onnx as donnx
from daceml.torch.module import dace_module
from examples.gnn_benchmark import sparse, models, util
from examples.gnn_benchmark.correctness import check_correctness
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.datasets import get_dataset
from examples.gnn_benchmark.util import stats_as_csv_entry, create_dace_model, \
    name_to_impl_class

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
    bwd_impl_name: str
    gnn_type: str
    convert_data: Callable[[torch_geometric.data.Data], sparse.GraphMatrix]
    model_eval: daceml.torch.DaceModule
    model_train: daceml.torch.DaceModule
    idx_dtype: torch.dtype
    val_dtype: torch.dtype
    correct: Optional[bool] = None
    correct_grads: Optional[bool] = None
    data: Optional[sparse.GraphMatrix] = None


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_model: torch.nn.Module,
                 torch_csr_args: Sequence[torch.Tensor],
                 torch_edge_list_args: Sequence[torch.Tensor],
                 args,
                 targets: Optional[torch.Tensor],
                 backward: bool,
                 save_output: bool = True,
                 small: bool = False,
                 skip_torch_csr: bool = False,
                 skip_torch_edge_list: bool = False):
    from examples.gnn_benchmark.performance_measurement import \
        print_time_statistics
    if use_gpu:
        from examples.gnn_benchmark.performance_measurement import \
            measure_performance
    else:
        from examples.gnn_benchmark.performance_measurement import \
            measure_performance_cpu as measure_performance

    experiment_infos = OrderedDict(experiment_infos)

    funcs = []
    func_names = []

    if not skip_torch_csr:
        torch_model.eval()
        funcs.append(lambda: torch_model(*torch_csr_args))
        func_names.append('torch_csr')
    if not skip_torch_edge_list:
        torch_model.eval()
        funcs.append(lambda: torch_model(*torch_edge_list_args))
        func_names.append('torch_edge_list')

    ### Forward pass.

    def run_with_inputs(model, inputs):
        return model(*inputs)

    for impl_spec, experiment_info in experiment_infos.items():
        model = experiment_info.model_eval
        inputs = experiment_info.data.to_input_list()

        funcs.append(functools.partial(run_with_inputs, model, inputs))
        name = "dace"
        if args.threadblock_dynamic:
            name += "_tb-dynamic"
        if args.no_opt:
            name += "_no_autoopt"
        if args.no_persistent_mem:
            name += "_no_persistent_mem"
        name += f"_{impl_spec}"
        func_names.append(name)

    print(f"---> Benchmarking the forward pass...")
    with torch.no_grad():
        grad_test = funcs[0]()
        assert grad_test.grad_fn is None
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

    ### Backward pass.
    if backward:
        assert targets is not None

        if hasattr(torch_model, 'conv2'):
            criterion = torch.nn.NLLLoss()
        else:
            criterion = lambda pred, targets: torch.sum(pred)

        def backward_fn(model, inputs):
            pred = model(*inputs)
            loss = criterion(pred, targets)
            loss.backward()

        backward_funcs = []
        if not skip_torch_csr:
            torch_model.train()
            backward_funcs.append(lambda: backward_fn(torch_model, torch_csr_args))
        if not skip_torch_edge_list:
            torch_model.train()
            backward_funcs.append(lambda: backward_fn(torch_model, torch_edge_list_args))

        for experiment_info in experiment_infos.values():
            model = experiment_info.model_train
            inputs = experiment_info.data.to_input_list()
            backward_funcs.append(functools.partial(backward_fn, model, inputs))

        print(f"---> Benchmarking the BACKWARD pass...")
        times = measure_performance(backward_funcs,
                                    func_names=func_names,
                                    warmups=5 if not small else 2,
                                    num_iters=5 if not small else 2,
                                    timing_iters=20 if not small else 3)
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
            # Write out some system info.
            file.write(
                f"# HOST {socket.gethostname()}, GPU {torch.cuda.get_device_name()}\n")

            headers = [
                'Name', 'Model', 'Size', 'Min', 'Mean',
                'Median',
                'Stdev', 'Max'
            ]
            file.write(','.join(headers) + '\n')
        file.write(
            stats_as_csv_entry(times, func_names, args.model,
                               args.hidden))


def parse_impl_spec(impl_spec: str):
    """We accept impl names of the form
    'impl_name-impl_arg1-impl_arg2:bwd_impl_name-bwd_impl_arg1...'
    where the bwd_impl_name and bwd_impl_args are optional. If no bwd_impl_name
    is specified, the forward impl is used for the backward pass.
    """
    if ':' in impl_spec:
        impl_spec, bwd_spec = impl_spec.split(':')
    else:
        bwd_spec = impl_spec
    impl_name, *impl_args = impl_spec.split('-')
    bwd_impl_name, *bwd_impl_args = bwd_spec.split('-')
    return impl_name, impl_args, bwd_impl_name, bwd_impl_args


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
    parser.add_argument('--no-persistent-mem', action='store_true')
    parser.add_argument('--no-opt', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys())
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    parser.add_argument('--idx-dtype', type=str, default='int32')
    parser.add_argument('--val-dtype', type=str, default='float32')
    parser.add_argument('--no-gen-code', action='store_true')
    parser.add_argument('--torch', choices=['both', 'csr', 'edge_list', 'none'], default='both')
    args = parser.parse_args()

    dtype_str_to_torch_type = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64
    }
    args.idx_dtype = dtype_str_to_torch_type[args.idx_dtype]
    args.val_dtype = dtype_str_to_torch_type[args.val_dtype]

    model_class = model_dict[args.model]
    num_hidden_features = args.hidden
    args.outfile = Path(args.outfile) if args.outfile is not None else None

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    data = get_dataset(args.data, device, val_dtype=args.val_dtype)

    print("Num node features: ", data.num_node_features)
    num_classes = data.y.max().item() + 1
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    print("Num nodes:", data.num_nodes)
    print("Num non zero:", data.num_edges)
    normalize = args.normalize
    print("Normalize: ", normalize)
    print("Implementation: ", args.impl)
    print("DaCe indices dtype: ", args.idx_dtype)
    print("DaCe values dtype: ", args.val_dtype)

    # Define models.
    model_args = (data.num_node_features, num_hidden_features, num_classes,
                  normalize)
    torch_model = model_class(*model_args, bias_init=torch.nn.init.uniform_).to(
        args.val_dtype).to(device)
    torch_model.eval()

    dace_models = OrderedDict()
    if args.impl == ['none']:
        args.impl = []
    elif args.impl == ['all']:
        args.impl = name_to_impl_class[args.model].keys()

    for impl_spec in args.impl:
        impl_name, impl_args, bwd_impl_name, bwd_impl_args = parse_impl_spec(
            impl_spec)
        implementation_class = name_to_impl_class[args.model][impl_name]
        if 'ellpack' in impl_name:
            block_size = impl_args[0]
            convert_data = functools.partial(
                implementation_class.convert_data,
                block_size=int(block_size))
        else:
            convert_data = implementation_class.convert_data
        dace_model_eval, dace_model_train = create_dace_model(torch_model,
                                                              sdfg_tag=impl_spec.replace(':', '_').replace('-', '_'),
                                                              implementation_name=impl_name,
                                                              backward_implementation_name=bwd_impl_name,
                                                              threadblock_dynamic=args.threadblock_dynamic,
                                                              persistent_mem=not args.no_persistent_mem,
                                                              do_opt=not args.no_opt,
                                                              device=device,
                                                              gen_code=not args.no_gen_code,
                                                              backward=args.backward)
        info = ExperimentInfo(impl_name=impl_name,
                              bwd_impl_name=bwd_impl_name,
                              model_eval=dace_model_eval,
                              model_train=dace_model_train,
                              convert_data=convert_data,
                              gnn_type=args.model,
                              idx_dtype=args.idx_dtype,
                              val_dtype=args.val_dtype)
        dace_models[impl_spec] = info

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
        torch_csr_args, torch_edge_list_args = None, None
        if args.torch == 'csr' or args.torch == 'both':
            torch_csr_args = util.make_torch_csr_args(data)
        if args.torch == 'edge_list' or args.torch == 'both':
            add_edge_weight = hasattr(data, 'edge_weight') and 'gcn' in args.model
            torch_edge_list_args = util.make_torch_edge_list_args(data, add_edge_weight)

        check_correctness(dace_models, torch_model,
                          torch_edge_list_args,
                          torch_csr_args,
                          targets=data.y,
                          backward=args.backward,
                          skip_torch_csr=args.torch != 'both' and args.torch != 'csr',
                          skip_torch_edge_list=args.torch != 'both' and args.torch != 'edge_list')

        if args.mode == 'benchmark' or args.mode == 'benchmark_small':
            do_benchmark(dace_models, torch_model, torch_csr_args,
                         torch_edge_list_args, args,
                         backward=args.backward,
                         targets=data.y,
                         save_output=True,
                         small=args.mode == 'benchmark_small',
                         skip_torch_csr=args.torch != 'both' and args.torch != 'csr',
                         skip_torch_edge_list=args.torch != 'both' and args.torch != 'edge_list')


if __name__ == '__main__':
    main()
