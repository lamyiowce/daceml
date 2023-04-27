import logging
from typing import Callable, Optional

import dace
import torch
from dace.dtypes import ScheduleType
from dace.transformation.auto.auto_optimize import \
    auto_optimize as dace_auto_optimize

import daceml


def _specialize_memory(sdfg):
    from dace.sdfg.scope import is_devicelevel_gpu

    arrays = []

    # Make memory persistent
    for state in sdfg.nodes():
        for dnode in state.data_nodes():
            if is_devicelevel_gpu(sdfg, state, dnode):
                continue
            if 'reserved_' in dnode.data:
                continue
            arr = sdfg.arrays[dnode.data]
            if (arr.transient and not isinstance(arr, dace.data.View)
                    and arr.storage != dace.StorageType.Register):
                if arr.lifetime != dace.AllocationLifetime.Persistent:
                    arrays.append(dnode.data)
                arr.lifetime = dace.AllocationLifetime.Persistent

    # Disable OpenMP sections
    sdfg.openmp_sections = False
    return arrays


def specialize_mem_onnx(mod):
    def spec(module):
        arrays = []
        for sd in module.sdfg.all_sdfgs_recursive():
            arrays += _specialize_memory(sd)
        print(
            f"Specialized {len(arrays)} arrays to persistent: {' '.join(arrays)}")

    mod.append_post_onnx_hook("specializemem", spec)


def apply_dace_auto_optimize(module):
    sdfg = module.sdfg
    dace_auto_optimize(
        sdfg,
        device=dace.dtypes.DeviceType.GPU
        if torch.cuda.is_available() else dace.dtypes.DeviceType.CPU)


def apply_dace_auto_opt_after_autodiff(forward_sdfg, backward_sdfg):
    dace_device = dace.dtypes.DeviceType.GPU if torch.cuda.is_available() else dace.dtypes.DeviceType.CPU
    dace_auto_optimize(forward_sdfg,
                       device=dace_device)
    dace_auto_optimize(backward_sdfg,
                       device=dace_device)


def make_maps_dynamic(module, exclude_loops=None):
    sdfg = module.sdfg
    # Count which loops were excluded to be able to produce a warning
    # in case a loop was not found in the graph.
    exclude_loops = {name: 0 for name in exclude_loops} or {}
    for node in sdfg.all_nodes_recursive():
        if isinstance(node[0], dace.sdfg.nodes.MapEntry) \
                and node[0].schedule == dace.dtypes.ScheduleType.Sequential \
                and len(node[0].map.params):
            if node[0].label not in exclude_loops:
                print("Changing schedule to TB dynamic: ", node[0].map)
                node[0].schedule = ScheduleType.GPU_ThreadBlock_Dynamic
            else:
                exclude_loops[node[0].label] += 1
                print("Keeping schedule sequential for ", node[0].map)

    not_excluded = [
        name for name, count in exclude_loops.items() if count == 0
    ]
    if not_excluded:
        logging.warning(
            "Following loops were marked as excluded from thread-block dynamic "
            "scheduling but were not found in the SDFG: %s", not_excluded)


def set_implementation(module: daceml.torch.module.DaceModule,
                       implementation_name: str,
                       backward_implementation_name: Optional[str] = None,
                       backward: bool = False):
    sdfg = module.sdfg
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node,
                      dace.sdfg.nodes.LibraryNode) and implementation_name in node.implementations:
            if backward:
                node.backward_implementation = backward_implementation_name
            node.implementation = implementation_name


def set_memory_to_register(sdfg: dace.SDFG, array_name: str):
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode) and node.data == array_name:
            arr = sdfg.arrays[node.data]
            arr.storage = dace.dtypes.StorageType.Register


def apply_to_both(fn: Callable[[dace.SDFG], None]):
    def wrapper(forward_sdfg, backward_sdfg):
        fn(forward_sdfg)
        fn(backward_sdfg)
    return wrapper

def set_reduce_to_gpuauto(sdfg: dace.SDFG):
    counter = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.LibraryNode) and isinstance(node, dace.libraries.standard.nodes.Reduce):
            print(f"Setting impl {node} from {node.implementation} to GPUAuto.")
            node.implementation = 'GPUAuto'
            counter += 1
    print(f"Set {counter} reduce nodes to GPUAuto")

