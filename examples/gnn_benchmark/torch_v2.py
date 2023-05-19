import argparse
import copy
import faulthandler
import logging
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.transforms import GCNNorm
from torch_geometric.nn.conv.cugraph import CuGraphModule
from torch_geometric.nn import FusedGATConv

import examples.gnn_benchmark.torch_util
from examples.gnn_benchmark import models
from examples.gnn_benchmark.benchmark import do_benchmark
from examples.gnn_benchmark.correctness import check_correctness
from examples.gnn_benchmark.datasets import get_dataset
from examples.gnn_benchmark.torch_profile import torch_profile

faulthandler.enable()
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
                  'gat': models.GAT, 'gcn_single_layer': models.GCNSingleLayer, 'cugraph_gat': models.CuGraphGAT,
                  'fused_gat': models.FusedGAT, 'gat_single_layer': models.GATSingleLayer}

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'dry',
                                           'benchmark_small', 'torch_profile'],
                        required=True)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--model', choices=model_dict.keys(), required=True)
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--idx-dtype', type=str, default='int32')
    parser.add_argument('--val-dtype', type=str, default='float32')
    parser.add_argument('--torch', choices=['all', 'csr', 'edge_list', 'none', 'compiled_edge_list', 'csc', 'csc_csr'], default='all')
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
    print("DaCe indices dtype: ", args.idx_dtype)
    print("DaCe values dtype: ", args.val_dtype)

    # Define models.
    torch_model = model_class(data.num_node_features, num_hidden_features, num_classes,
                              bias_init=torch.nn.init.uniform_).to(
        args.val_dtype).to(device)
    torch_model.eval()
    print(torch_model)

    # Normalize data for GCN.
    if 'gcn' in args.model:
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    torch_experiments = []
    if args.torch == 'csc':
        assert args.model == 'cugraph_gat'
        torch_csc_args = CuGraphModule.to_csc(data.edge_index)
        torch_experiments += [('torch_csc', torch_model, (data.x, torch_csc_args))]
    if args.torch == 'csc_csr':
        assert args.model == 'fused_gat'
        torch_args = FusedGATConv.to_graph_format(data.edge_index)
        torch_experiments += [('torch_dgnn', torch_model, (data.x, *torch_args))]
    if args.torch == 'edge_list' or args.torch == 'all':
        add_edge_weight = hasattr(data, 'edge_weight') and 'gcn' in args.model
        torch_edge_list_args = examples.gnn_benchmark.torch_util.make_torch_edge_list_args(
            data, add_edge_weight)
        torch_experiments += [
            ('torch_edge_list', torch_model, torch_edge_list_args)]
    if args.torch in ['compiled_edge_list', 'all']:
        compiled_torch_model = torch_geometric.compile(copy.deepcopy(torch_model))
        compiled_torch_model.eval()
        if args.torch == 'compiled_edge_list':
            del torch_model
        add_edge_weight = hasattr(data, 'edge_weight') and 'gcn' in args.model
        torch_edge_list_args = examples.gnn_benchmark.torch_util.make_torch_edge_list_args(
            data, add_edge_weight)
        torch_experiments += [
            ('compiled_torch_edge_list', compiled_torch_model, torch_edge_list_args)]
    if args.torch == 'csr' or args.torch == 'all':
        torch_csr_args = examples.gnn_benchmark.torch_util.make_torch_csr_args(
            data)
        torch_experiments += [('torch_csr', torch_model, torch_csr_args)]

    if 'single_layer' in args.model:
        loss_fn = lambda pred, targets: torch.sum(pred)
    else:
        loss_fn = torch.nn.NLLLoss()

    check_correctness(dace_models=dict(),
                      torch_experiments=torch_experiments,
                      loss_fn=loss_fn,
                      targets=data.y,
                      backward=args.backward)
    if args.mode == 'benchmark' or args.mode == 'benchmark_small':
        do_benchmark(dict(),
                     torch_experiments=torch_experiments,
                     backward=args.backward,
                     loss_fn=loss_fn,
                     targets=data.y,
                     hidden_size=args.hidden,
                     model_name=args.model,
                     dace_tag=None,
                     use_gpu=use_gpu,
                     outfile=args.outfile,
                     small=args.mode == 'benchmark_small')
    elif args.mode == 'torch_profile':
        torch_profile(dict(),
                      torch_model,
                      torch_csr_args,
                      torch_edge_list_args,
                      args,
                      backward=args.backward,
                      targets=data.y,
                      skip_torch_csr=args.torch != 'all' and args.torch != 'csr',
                      skip_torch_edge_list=args.torch != 'all' and args.torch != 'edge_list')
    elif args.mode != 'dry':
        raise ValueError(f"Invalid mode {args.mode}.")


if __name__ == '__main__':
    main()
