import argparse
import faulthandler
import functools
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.transforms import GCNNorm

import examples.gnn_benchmark.torch_util
from daceml import onnx as donnx
from examples.gnn_benchmark import models
from examples.gnn_benchmark.benchmark import do_benchmark
from examples.gnn_benchmark.correctness import check_correctness
from examples.gnn_benchmark.data_optimizer import optimize_data
from examples.gnn_benchmark.datasets import get_dataset
from examples.gnn_benchmark.experiment_info import ExperimentInfo
from examples.gnn_benchmark.torch_profile import torch_profile
from examples.gnn_benchmark.util import name_to_impl_class, create_dace_model

faulthandler.enable()
donnx.default_implementation = "pure"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(42)
np.random.seed(42)
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


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
                  'gat': models.GAT, 'gcn_single_layer': models.GCNSingleLayer,
                  'gat_single_layer': models.GATSingleLayer}

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'dry',
                                           'benchmark_small', 'torch_profile'],
                        required=True)
    parser.add_argument('--impl', type=str, nargs='+', required=True)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--no-persistent-mem', action='store_true')
    parser.add_argument('--no-opt', action='store_true')
    parser.add_argument('--zero-bias', action='store_true')
    parser.add_argument('--no-bias', action='store_true')
    parser.add_argument('--threadblock-dynamic', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys(), required=True)
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--name', type=str, default='dace')
    parser.add_argument('--idx-dtype', type=str, default='int32')
    parser.add_argument('--val-dtype', type=str, default='float32')
    parser.add_argument('--no-gen-code', action='store_true')
    parser.add_argument('--torch', choices=['both', 'csr', 'edge_list', 'none'], default='both')
    parser.add_argument('--tag', type=str, default=None)
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
    bias_init_fn = torch.nn.init.uniform_ if not args.zero_bias else torch.nn.init.zeros_

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
    print("Bias: ", not args.no_bias)
    print("Bias init fn: ", bias_init_fn)
    print("Implementation: ", args.impl)
    print("DaCe indices dtype: ", args.idx_dtype)
    print("DaCe values dtype: ", args.val_dtype)
    print("CUDA available: ", torch.cuda.is_available())

    # Define models.
    additional_kwargs = {} if 'gcn' in args.model else {'num_heads': args.heads}
    torch_model = model_class(data.num_node_features, num_hidden_features, num_classes,
                              bias_init=bias_init_fn, bias=not args.no_bias, **additional_kwargs)
    torch_model = torch_model.to(
        args.val_dtype).to(device)
    torch_model.eval()

    dace_models = create_experiments(args, torch_model)

    # Normalize data for GCN.
    if 'gcn' in args.model:
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    torch_model, dace_models = optimize_data(torch_model, dace_models, data)

    for k, v in dace_models.items():
        print(f"Impl: {k}")
        print(v)

    torch_experiments = []
    torch_csr_args, torch_edge_list_args = None, None
    if args.torch == 'csr' or args.torch == 'both':
        torch_csr_args = examples.gnn_benchmark.torch_util.make_torch_csr_args(data)
        torch_experiments += [('torch_csr', torch_model, torch_csr_args)]
    if args.torch == 'edge_list' or args.torch == 'both':
        add_edge_weight = hasattr(data, 'edge_weight') and 'gcn' in args.model
        torch_edge_list_args = examples.gnn_benchmark.torch_util.make_torch_edge_list_args(data,
                                                                                           add_edge_weight)
        torch_experiments += [('torch_edge_list', torch_model, torch_edge_list_args)]

    if 'single_layer' in args.model:
        loss_fn = lambda pred, targets: torch.sum(pred)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    check_correctness(dace_models,
                      torch_experiments=torch_experiments,
                      loss_fn=loss_fn,
                      targets=data.y,
                      backward=args.backward)

    if args.mode == 'benchmark' or args.mode == 'benchmark_small':
        dace_tag = "dace"
        if args.threadblock_dynamic:
            dace_tag += "_tb-dynamic"
        if args.no_opt:
            dace_tag += "_no_autoopt"
        if args.no_persistent_mem:
            dace_tag += "_no_persistent_mem"
        if args.tag is not None:
            dace_tag += f"_{args.tag}"

        do_benchmark(dace_models,
                     backward=args.backward,
                     loss_fn=loss_fn,
                     targets=data.y,
                     hidden_size=args.hidden,
                     model_name=args.model,
                     dace_tag=dace_tag,
                     use_gpu=use_gpu,
                     outfile=args.outfile,
                     small=args.mode == 'benchmark_small',
                     torch_experiments=torch_experiments,
                     )
    elif args.mode == 'torch_profile':
        torch_profile(dace_models,
                      torch_model,
                      torch_csr_args,
                      torch_edge_list_args,
                      args,
                      backward=args.backward,
                      targets=data.y,
                      skip_torch_csr=args.torch != 'both' and args.torch != 'csr',
                      skip_torch_edge_list=args.torch != 'both' and args.torch != 'edge_list')
    elif args.mode != 'dry':
        raise ValueError(f"Invalid mode {args.mode}.")


def create_experiments(args, torch_model):
    dace_models = OrderedDict()

    impl_specs = args.impl
    if impl_specs == ['none']:
        impl_specs = []
    elif impl_specs == ['all']:
        impl_specs = name_to_impl_class[args.model].keys()
    for impl_spec in impl_specs:
        impl_name, impl_args, bwd_impl_name, bwd_impl_args = parse_impl_spec(
            impl_spec)
        implementation_class = name_to_impl_class[args.model][impl_name]
        format_args = implementation_class.graph_format.parse_args(impl_args)
        sdfg_tag = impl_spec.replace(':', '__').replace('-', '_').replace('.', '_')

        dace_model_eval, dace_model_train = create_dace_model(torch_model,
                                                              sdfg_tag=sdfg_tag,
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
                              graph_format=implementation_class.graph_format,
                              graph_format_args=format_args,
                              gnn_type=args.model,
                              idx_dtype=args.idx_dtype,
                              val_dtype=args.val_dtype)
        dace_models[impl_spec] = info

    return dace_models


if __name__ == '__main__':
    main()
