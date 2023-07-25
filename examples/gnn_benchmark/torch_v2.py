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
from examples.gnn_benchmark.common import get_loss_and_targets
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


def copy_and_set_grad(tensor: torch.Tensor, requires_grad):
    tensor = tensor.detach().clone(memory_format=torch.contiguous_format)
    tensor.requires_grad = requires_grad
    return tensor


def main():
    model_dict = {'gcn': models.GCN,
                  'linear': models.LinearModel,
                  'gat': models.GAT,
                  'gcn_single_layer': models.GCNSingleLayer,
                  'gat_single_layer': models.GATSingleLayer}

    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--data', required=True)
    parser.add_argument('--mode', choices=['benchmark', 'dry',
                                           'benchmark_small', 'torch_profile'],
                        required=True)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--input-grad', action='store_true')
    parser.add_argument('--force-num-features', type=int, default=None)
    parser.add_argument('--model', choices=model_dict.keys(), required=True)
    parser.add_argument('--hidden', type=int, default=None, required=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--idx-dtype', type=str, default='int32')
    parser.add_argument('--val-dtype', type=str, default='float32')
    parser.add_argument('--torch', type=str, required=True)
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

    data = get_dataset(args.data, device, val_dtype=args.val_dtype,
                       force_num_features=args.force_num_features)

    print("Num node features: ", data.num_node_features)
    num_classes = data.y.max().item() + 1
    print("Num classes: ", num_classes)
    print("Num hidden features: ", num_hidden_features)
    print("Num nodes:", data.num_nodes)
    print("Num non zero:", data.num_edges)
    print("DaCe indices dtype: ", args.idx_dtype)
    print("DaCe values dtype: ", args.val_dtype)

    # Normalize data for GCN.
    if 'gcn' in args.model:
        gcn_norm = GCNNorm(add_self_loops=True)
        data = gcn_norm(data)

    model_kwargs = {}

    if 'cugraph' in args.torch:
        from torch_geometric.nn import CuGraphGATConv
        assert 'gat' in args.model
        graph_inputs = CuGraphModule.to_csc(data.edge_index)
        inputs = (copy_and_set_grad(data.x, requires_grad=args.input_grad), *graph_inputs)
        model_kwargs = {'gat_layer': CuGraphGATConv}
    elif 'dgnn' in args.torch:
        assert 'gat' in args.model
        from torch_geometric.nn import FusedGATConv
        graph_inputs = FusedGATConv.to_graph_format(data.edge_index)
        inputs = (copy_and_set_grad(data.x, requires_grad=args.input_grad), *graph_inputs)
        model_kwargs = {'gat_layer': FusedGATConv}
    elif 'edge_list' in args.torch:
        add_edge_weight = hasattr(data, 'edge_weight') and 'gcn' in args.model
        inputs = examples.gnn_benchmark.torch_util.make_torch_edge_list_args(
            data, add_edge_weight, input_grad=args.input_grad)
    elif 'csr' in args.torch:
        inputs = examples.gnn_benchmark.torch_util.make_torch_csr_args(
            data, input_grad=args.input_grad)
    else:
        raise ValueError(f'Unknown torch impl {args.torch}.')

    torch_model = model_class(data.num_node_features, num_hidden_features,
                              num_classes, args.num_layers,
                              bias_init=torch.nn.init.uniform_,
                              **model_kwargs)

    torch_model = torch_model.to(args.val_dtype).to(device)
    torch_model.eval()
    print(torch_model)

    if 'compiled' in args.torch:
        torch_model = torch_geometric.compile(torch_model)
        torch_model.eval()

    experiment = [(f'torch_{args.torch}', torch_model, inputs)]

    print("Running experiment: ", experiment)

    loss_fn, targets = get_loss_and_targets(args.model, args.val_dtype, data, num_classes)

    check_correctness(dace_models=dict(),
                      torch_experiments=experiment,
                      loss_fn=loss_fn,
                      targets=targets,
                      backward=args.backward)
    if args.mode == 'benchmark' or args.mode == 'benchmark_small':
        do_benchmark(dict(),
                     torch_experiments=experiment,
                     backward=args.backward,
                     loss_fn=loss_fn,
                     targets=targets,
                     hidden_size=args.hidden,
                     model_name=args.model,
                     dace_tag=None,
                     use_gpu=use_gpu,
                     outfile=args.outfile,
                     small=args.mode == 'benchmark_small',
                     num_layers=args.num_layers,
                     in_features_size=data.num_node_features)
    # elif args.mode == 'torch_profile':
    #     torch_profile(dict(),
    #                   torch_model,
    #                   torch_csr_args,
    #                   torch_edge_list_args,
    #                   args,
    #                   backward=args.backward,
    #                   targets=data.y,
    #                   skip_torch_csr=args.torch != 'all' and args.torch != 'csr',
    #                   skip_torch_edge_list=args.torch != 'all' and args.torch != 'edge_list')
    elif args.mode != 'dry':
        raise ValueError(f"Invalid mode {args.mode}.")


if __name__ == '__main__':
    main()
