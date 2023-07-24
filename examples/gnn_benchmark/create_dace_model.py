import copy
import functools
from typing import Tuple, Callable, Optional

import dace
import torch

from daceml.torch import DaceModule
from examples.gnn_benchmark import sdfg_util, models
from examples.gnn_benchmark.sdfg_util import make_maps_dynamic, apply_dace_auto_optimize, \
    specialize_mem_onnx


def create_dace_model(model: torch.nn.Module,
                      sdfg_tag: str,
                      implementation_name: str,
                      backward_implementation_name: str,
                      do_opt: bool,
                      persistent_mem: bool,
                      threadblock_dynamic: bool,
                      device: torch.device,
                      backward: bool,
                      compute_input_grad: bool,
                      gen_code: bool = True,
                      ) -> Tuple[dace.DaceModule, dace.DaceModule]:
    sdfg_name = f"{model.__class__.__name__}_{sdfg_tag}"

    # Hack: model inputs will have names 0, 1, 2, 3... and we want to skip
    # calculating the gradients for all of them.
    inputs_to_skip = [str(i) for i in range(20)]
    if compute_input_grad:
        # Compute input grad for input features only.
        inputs_to_skip = inputs_to_skip[1:]
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

    if persistent_mem:
        print("---> Adding persistent memory hook.")
        specialize_mem_onnx(dace_model)

    print("/////////////////////")
    print(">>> Model hooks:")
    print("> Post onnx:")
    for name in dace_model.post_onnx_hooks:
        print(name)
    print("> Post autodiff:")
    for name in dace_model.post_autodiff_hooks:
        print(name)
    print("/////////////////////")
