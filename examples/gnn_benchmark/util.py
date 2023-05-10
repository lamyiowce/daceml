import copy
import functools
import statistics
from typing import Dict, Tuple, Optional

import dace
import torch
from torch_sparse import SparseTensor

from daceml.onnx import register_replacement, TORCH_DTYPE_TO_TYPECLASS
from daceml.onnx.nodes import replacement_entries

from daceml.torch import DaceModule
from examples.gnn_benchmark import models, sdfg_util
from examples.gnn_benchmark.implementations import gcn_implementations, \
    gat_implementations
from examples.gnn_benchmark.implementations.common import SparseLayerBase, \
    SpecialInputType
from examples.gnn_benchmark.sdfg_util import apply_dace_auto_optimize, \
    specialize_mem_onnx, make_maps_dynamic, apply_dace_auto_opt_after_autodiff

name_to_impl_class: Dict[str, Dict[str, SparseLayerBase]] = {
    "gcn": {"csr": gcn_implementations.GCNConvCSR,
            "csr_reorder": gcn_implementations.GCNConvCSRReordered,
            "coo": gcn_implementations.GCNConvCOO,
            "csc": gcn_implementations.GCNConvCSC,
            "ellpack_t": gcn_implementations.GCNConvEllpackTransposed,
            "ellpack": gcn_implementations.GCNConvEllpack,
            "semester_thesis": gcn_implementations.GCNConvSemesterThesis},
    "gat": {"csr": gat_implementations.GATConvCSR,
            "semester_thesis": gat_implementations.GATConvSemesterThesis}
}
name_to_impl_class['gcn_single_layer'] = name_to_impl_class['gcn']


def stats_as_csv_entry(times, func_names, model, hidden_size):
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


def create_dace_model(model: torch.nn.Module,
                      sdfg_tag: str,
                      implementation_name: str,
                      backward_implementation_name: str,
                      do_opt: bool,
                      persistent_mem: bool,
                      threadblock_dynamic: bool,
                      device: torch.device,
                      backward: bool,
                      gen_code: bool = True
                      ) -> Tuple[dace.DaceModule, dace.DaceModule]:
    sdfg_name = f"{model.__class__.__name__}_{sdfg_tag}"

    dace_model_eval = DaceModule(copy.deepcopy(model),
                                 sdfg_name=sdfg_name + "_eval",
                                 backward=False,
                                 regenerate_code=gen_code,
                                 inputs_to_skip=['0', '1', '2', '3']).to(device)
    add_hooks(dace_model_eval, backward=False, device=device, do_opt=do_opt,
              implementation_name=implementation_name,
              backward_implementation_name=backward_implementation_name,
              persistent_mem=persistent_mem,
              threadblock_dynamic=threadblock_dynamic)

    dace_model_train = None
    if backward:
        dace_model_train = DaceModule(copy.deepcopy(model),
                                      sdfg_name=sdfg_name + "_train",
                                      backward=True,
                                      regenerate_code=gen_code,
                                      inputs_to_skip=['0', '1', '2', '3']).to(
            device)
        add_hooks(dace_model_train, backward=True, device=device, do_opt=do_opt,
                  implementation_name=implementation_name,
                  backward_implementation_name=backward_implementation_name,
                  persistent_mem=persistent_mem,
                  threadblock_dynamic=threadblock_dynamic)

    return dace_model_eval, dace_model_train


def add_hooks(dace_model: DaceModule, backward: bool, device: torch.device,
              do_opt: bool, implementation_name: str,
              backward_implementation_name: Optional[str],
              persistent_mem: bool, threadblock_dynamic: bool):
    if device.type == 'cuda':
        set_reduce_implementation = functools.partial(
            sdfg_util.set_reduce_implementation, implementation_name='GPUAuto')
        dace_model.append_post_onnx_hook("set_reduce_implementation_post_onnx",
                                         lambda model: set_reduce_implementation(model.sdfg))

        dace_model.append_post_autodiff_hook("set_reduce_implementation_post_autodiff",
                                             sdfg_util.apply_to_both(set_reduce_implementation))
    if persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)
    if threadblock_dynamic:
        print("---> Adding threadblock dynamic maps hook.")
        exclude_loops = []
        if isinstance(dace_model.model, models.GCN):
            # Has to be skipped, otherwise the computation results are incorrect.
            exclude_loops = [
                'daceml_onnx_op_implementations_replacement_implementations_prog_sparse_45_4_46'
            ]
        elif isinstance(dace_model.model, models.GAT):
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
        sdfg_util.set_implementation,
        implementation_name=implementation_name,
        backward_implementation_name=backward_implementation_name,
        backward=backward)
    dace_model.prepend_post_onnx_hook("set_implementation",
                                      set_implementation)
    if do_opt:
        print("---> Adding auto-opt hook.")
        if backward:
            dace_model.append_post_autodiff_hook("dace_auto_optimize",
                                                 apply_dace_auto_opt_after_autodiff)
        else:
            dace_model.append_post_onnx_hook("dace_auto_optimize",
                                             apply_dace_auto_optimize)
    fn = lambda forward_sdfg, backward_sdfg: sdfg_util.set_memory_to_register(
        backward_sdfg, '__tmp3')
    dace_model.append_post_autodiff_hook("Set __tmp3 to register", fn)

    def simplify(sdfg: dace.SDFG):
        sdfg.simplify(verbose=True)

    dace_model.append_post_autodiff_hook("simplify",
                                         sdfg_util.apply_to_both(simplify))


def register_replacement_overrides(implementation_name, layer_name, idx_dtype,
                                   val_dtype):
    impl_class = name_to_impl_class[layer_name][implementation_name]
    input_spec = impl_class.input_spec
    if idx_dtype not in impl_class.allowed_idx_dtypes:
        raise ValueError(
            f"idx_dtype {idx_dtype} not allowed for {layer_name} with {implementation_name}. Allowed: {impl_class.allowed_idx_dtypes}")
    idx_dtype = TORCH_DTYPE_TO_TYPECLASS[idx_dtype]
    val_dtype = TORCH_DTYPE_TO_TYPECLASS[val_dtype]
    map_dtype = {SpecialInputType.IDX_DTYPE: idx_dtype,
                 SpecialInputType.VAL_DTYPE: val_dtype}
    input_spec = {k: map_dtype.get(v, v) for k, v in input_spec.items()}
    if 'gcn' in layer_name:
        register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                             inputs=input_spec,
                             outputs={'output': val_dtype},
                             shape_infer=replacement_entries.shape_infer_GCNConv,
                             shape_fn_from_module=replacement_entries.make_GCNConv_shape_fn)
    elif layer_name == 'gat':
        register_replacement('torch_geometric.nn.conv.gat_conv.GATConv',
                             inputs=input_spec,
                             outputs={'output': val_dtype},
                             shape_infer=replacement_entries.shape_infer_GATConv,
                             shape_fn_from_module=replacement_entries.make_GATConv_shape_fn)


def make_torch_edge_list_args(data, add_edge_weights):
    '''Create an argument list for the torch edge list model.'''
    torch_edge_list_args = data.x.contiguous(), data.edge_index.contiguous()
    if add_edge_weights:
        torch_edge_list_args += (data.edge_weight.contiguous(),)
    return torch_edge_list_args


def make_torch_csr_args(data):
    """Create argument lists for torch CSR models."""
    sparse_edge_index = SparseTensor.from_edge_index(
        data.edge_index, edge_attr=data.edge_weight)

    # pyg requires the sparse tensor input to be transposed.
    torch_csr_args = data.x.contiguous(), sparse_edge_index.t()
    return torch_csr_args
