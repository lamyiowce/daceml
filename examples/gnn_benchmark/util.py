import copy
import functools
from typing import Dict, Tuple, Optional, Callable

import dace
import torch

from daceml.onnx import register_replacement, TORCH_DTYPE_TO_TYPECLASS
from daceml.onnx.nodes import replacement_entries
from daceml.torch import DaceModule
from examples.gnn_benchmark import models, sdfg_util
from examples.gnn_benchmark.implementations import gcn_implementations, \
    gat_implementations
from examples.gnn_benchmark.implementations.common import SparseLayerBase, \
    SpecialInputType
from examples.gnn_benchmark.sdfg_util import apply_dace_auto_optimize, \
    specialize_mem_onnx, make_maps_dynamic, change_map_schedule, \
    simplify_hook, set_library_node_implementation

name_to_impl_class: Dict[str, Dict[str, SparseLayerBase]] = {
    "gcn": {
        "csr": gcn_implementations.GCNConvCSR,
        "csr_adapt": gcn_implementations.GCNConvCSRAdapt,
        "coo": gcn_implementations.GCNConvCOO,
        "coo_adapt": gcn_implementations.GCNConvCOOAdapt,
        "coo_cached": gcn_implementations.GCNConvCOOCached,
        "csc": gcn_implementations.GCNConvCSC,
        "csc_adapt": gcn_implementations.GCNConvCSCAdapt,
        "ellpack_t": gcn_implementations.GCNConvEllpackTransposed,
        "ellpack": gcn_implementations.GCNConvEllpack,
        "semester_thesis": gcn_implementations.GCNConvSemesterThesis,
        "csr_coo": gcn_implementations.GCNConvCSRCOO,
        "csr_coo_adapt": gcn_implementations.GCNConvCSRCOOAdapt,
    },
    "gat": {
        "semester_thesis": gat_implementations.GATConvSemesterThesis,
        "csr": gat_implementations.GATConvCSR,
        "coo": gat_implementations.GATConvCOO,
        "coo_stable": gat_implementations.GATConvCOOStable,
        "coo_cached": gat_implementations.GATConvCOOCached,
        "coo_stable_cached": gat_implementations.GATConvCOOStableCached,
        "csr_stable": gat_implementations.GATConvCSRStable,
        "csc": gat_implementations.GATConvCSC,
    }
}
name_to_impl_class['gcn_single_layer'] = name_to_impl_class['gcn']
name_to_impl_class['gat_single_layer'] = name_to_impl_class['gat']


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

    # Hack: model inputs will have names 0, 1, 2, 3... and we want to skip
    # calculating the gradients for all of them.
    inputs_to_skip = [str(i) for i in range(20)]
    dace_model_eval = DaceModule(copy.deepcopy(model),
                                 sdfg_name=sdfg_name + "_eval",
                                 backward=False,
                                 regenerate_code=gen_code,
                                 inputs_to_skip=inputs_to_skip).to(device)
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
                                      inputs_to_skip=inputs_to_skip).to(
            device)
        add_hooks(dace_model_train, backward=True, device=device, do_opt=do_opt,
                  implementation_name=implementation_name,
                  backward_implementation_name=backward_implementation_name,
                  persistent_mem=persistent_mem,
                  threadblock_dynamic=threadblock_dynamic)

    return dace_model_eval, dace_model_train


def add_hook(dace_model: DaceModule, name: str, fn: Callable, backward: bool):
    if not backward:
        dace_model.append_post_onnx_hook(name + "_post_onnx",
                                         lambda model: fn(model.sdfg))
    else:
        dace_model.append_post_autodiff_hook(name + "_post_autodiff",
                                             sdfg_util.apply_to_both(fn))


def add_hooks(dace_model: DaceModule, backward: bool, device: torch.device,
              do_opt: bool, implementation_name: str,
              backward_implementation_name: Optional[str],
              persistent_mem: bool, threadblock_dynamic: bool):
    if device.type == 'cuda':
        set_reduce_implementation = functools.partial(
            sdfg_util.set_reduce_implementation, implementation_name='GPUAuto')
        add_hook(dace_model, "set_reduce_implementation",
                 set_reduce_implementation, backward)

        if backward:
            fn = functools.partial(sdfg_util.change_storage,
                                   array_name=r'ONNX_\d+',
                                   new_storage=dace.StorageType.GPU_Global)
            add_hook(dace_model, "Set ONNX to GPU_global", fn, backward=False)

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

    # if device.type == 'cuda':

    # add_hook(dace_model, "make_outer_map_seq", change_out_map_schedule_fn, backward)

    # set_coomm_implementation = functools.partial(
    #     sdfg_util.set_library_node_implementation, implementation_name='cuSPARSE',
    #     node_name='COOMM',
    #     schedule=dace.dtypes.ScheduleType.GPU_Device)
    #
    # add_hook(dace_model, "set_coomm_implementation", set_coomm_implementation, backward)
    #
    # set_csrmm_implementation = functools.partial(
    #     sdfg_util.set_library_node_implementation, implementation_name='cuSPARSE',
    #     node_name='CSRMM',
    #     schedule=dace.dtypes.ScheduleType.GPU_Device)
    #
    # add_hook(dace_model, "set_csrmm_implementation", set_csrmm_implementation, backward)

    if do_opt:
        print("---> Adding auto-opt hook.")
        add_hook(dace_model, "dace_auto_optimize", apply_dace_auto_optimize,
                 backward)

    if persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)

    fn = lambda forward_sdfg, backward_sdfg: sdfg_util.change_storage(
        backward_sdfg, '__tmp3')
    dace_model.append_post_autodiff_hook("Set __tmp3 to register", fn)

    fn = functools.partial(sdfg_util.change_storage,
                           array_name='__tmp1',
                           expected_shape=(1, 1))
    add_hook(dace_model, "Set __tmp1 to register", fn, backward)

    # For GAT backward COO.
    fn = functools.partial(sdfg_util.change_storage,
                           array_name='__tmp\d+',
                           expected_shape=(1,))
    add_hook(dace_model, "Set __tmp0 to register", fn, backward)
    fn = functools.partial(sdfg_util.change_storage,
                           array_name='.+_mult',
                           expected_shape=(1,))
    add_hook(dace_model, "Set mults to register", fn, backward)

    # dace_model.append_post_onnx_hook("Set __tmp1 to register", fn)
    # dace_model.append_post_autodiff_hook("Set __tmp1 to register", fn)

    # def set_node_to_persistent(sdfg: dace.SDFG, array_name: str = "__tmp7"):
    #     for node, subsdfg in sdfg.all_nodes_recursive():
    #         if isinstance(node, dace.nodes.AccessNode) and node.data == array_name:
    #             arr = subsdfg.arrays[node.data]
    #             print(f"Setting Lifetime for {node} to persistent.")
    #             arr.storage = dace.dtypes.AllocationLifetime.Persistent
    # add_hook(dace_model, "set_tmp7_to_persistent", set_node_to_persistent, backward)

    # add_hook(dace_model, "simplify", simplify_hook, backward=backward)

    add_hook(dace_model, "flatten_blocks_for_1d_maps",
             sdfg_util.flatten_blocks_for_1d_maps,
             backward=backward)

    if device.type == 'cuda':
        fn = functools.partial(sdfg_util.change_storage,
                               array_name=r'examples_gnn_benchmark_implementations_gat_backward_backward_fn_\d+_\d+___tmp\d\d',
                               expected_shape=(1,))
        add_hook(dace_model,
                 "Move 'examples_gnn_benchmark_implementations_gat_backward_backward_fn_128_4___tmp32' to register",
                 fn, backward)
        fn = functools.partial(sdfg_util.change_storage,
                               array_name=r'examples_gnn_benchmark_implementations_gat_backward_basic_gat_backward_\d+_\d+___tmp\d\d',
                               expected_shape=(1,))
        add_hook(dace_model,
                 "Move 'examples_gnn_benchmark_implementations_gat_backward_basic_gat_backward_142_8___tmp18' to register",
                 fn, backward)

    print("/////////////////////")
    print(">>> Model hooks:")
    print("> Post onnx:")
    for name in dace_model.post_onnx_hooks:
        print(name)
    print("> Post autodiff:")
    for name in dace_model.post_autodiff_hooks:
        print(name)
    print("/////////////////////")


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
    output_spec = impl_class.output_spec
    output_spec = {k: map_dtype.get(v, v) for k, v in output_spec.items()}
    buffer_spec = impl_class.buffer_spec
    symbolic_override_fn = impl_class.ssi_fn
    if 'gcn' in layer_name:
        register_replacement('torch_geometric.nn.conv.gcn_conv.GCNConv',
                             inputs=input_spec,
                             outputs=output_spec,
                             shape_infer=symbolic_override_fn,
                             shape_fn_from_module=replacement_entries.make_GCNConv_shape_fn,
                             buffer_specs=buffer_spec)
    elif 'gat' in layer_name:
        register_replacement('torch_geometric.nn.conv.gat_conv.GATConv',
                             inputs=input_spec,
                             outputs=output_spec,
                             shape_infer=symbolic_override_fn,
                             shape_fn_from_module=replacement_entries.make_GATConv_shape_fn,
                             buffer_specs=buffer_spec)
    else:
        raise ValueError("Unknown layer name, no replacement registered.")
