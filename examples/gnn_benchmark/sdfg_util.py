import functools
import logging
import re
from typing import Callable, Optional, Tuple, List

import dace
import torch
from dace import Config
from dace.dtypes import ScheduleType
from dace.sdfg import nodes as nd

import daceml
from examples.gnn_benchmark.my_auto_optimize import my_auto_optimize


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
                    arrays.append((dnode.data, arr.lifetime))
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
            f"Specialized {len(arrays)} arrays to persistent: {' '.join([str(x) for x in arrays])}")

    mod.append_post_onnx_hook("specializemem", spec)


def simplify_hook(sdfg: dace.SDFG):
    sdfg.simplify(verbose=True)


def apply_dace_auto_optimize(sdfg):
    my_auto_optimize(
        sdfg,
        device=dace.dtypes.DeviceType.GPU
        if torch.cuda.is_available() else dace.dtypes.DeviceType.CPU)


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


def set_memory_to_register(sdfg: dace.SDFG, array_name: str,
                           expected_shape: Tuple[int, ...] = None):
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode) and node.data == array_name:
            arr = sdfg.arrays[node.data]
            if expected_shape is None or arr.shape == expected_shape:
                print(f"Setting storage for {node} to register.")
                arr.storage = dace.dtypes.StorageType.Register


def apply_to_both(fn: Callable[[dace.SDFG], None]):
    def wrapper(forward_sdfg, backward_sdfg):
        fn(forward_sdfg)
        fn(backward_sdfg)

    return wrapper


def set_library_node_implementation(sdfg: dace.SDFG,
                                   implementation_name: str,
                                   node_name: str = None,
                                   node_class: type = None,
                                   schedule: dace.dtypes.ScheduleType = None):
    def matches(node: nd.LibraryNode):
        if node_name is not None and node.label != node_name:
            return False
        if node_class is not None and not isinstance(node, node_class):
            return False
        return True

    counter = 0
    counter_already = 0

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nd.LibraryNode) and matches(node):
            if node.implementation == implementation_name:
                counter_already += 1
            else:
                print(
                    f"📚  Setting impl {node} ({node.schedule}) from {node.implementation} to {implementation_name}.")
                node.implementation = implementation_name
                node.schedule = schedule or node.schedule
                counter += 1

    print(
        f"📚 📚 📚  Set {counter} nodes of {node_name} / {node_class}  to {implementation_name}, {counter_already} were already set.")


def set_reduce_implementation(sdfg: dace.SDFG, implementation_name: str = 'GPUAuto'):
    set_library_node_implementation(sdfg, implementation_name, node_class=dace.libraries.standard.nodes.Reduce)


def get_tb_maps_recursive(subgraph):
    res = []
    for node in subgraph.nodes():
        if isinstance(node, nd.NestedSDFG):
            for state in node.sdfg.states():
                tbmaps = get_tb_maps_recursive(state)
                for map, sym_map in tbmaps:
                    for k in sym_map.values():
                        for kk, vv in node.symbol_mapping.items():
                            sym_map[k] = sym_map[k].subs(dace.symbol(kk), vv)
                    res.append((map, sym_map))
        elif isinstance(node, nd.MapEntry) and node.schedule in (
                dace.dtypes.ScheduleType.GPU_Device,
                dace.dtypes.ScheduleType.GPU_ThreadBlock,
                dace.dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
        ):
            res.append(
                (node.map, {dace.symbol(k): dace.symbol(k) for k in node.map.range.free_symbols}))
    return res


def flatten_blocks_for_1d_maps(sdfg: dace.SDFG):
    default_block_size = [int(b) for b in
                          Config.get('compiler', 'cuda', 'default_block_size').split(',')]
    total_size = functools.reduce(lambda a, b: a * b, default_block_size)

    for node, dfg_scope in sdfg.all_nodes_recursive():
        if isinstance(node, nd.MapEntry) \
                and node.schedule == dace.dtypes.ScheduleType.GPU_Device \
                and len(node.map.params):
            if len(node.map.params) == 1 and node.map.gpu_block_size is None:
                subgraph = dfg_scope.scope_subgraph(node)
                sub_maps = get_tb_maps_recursive(subgraph)
                if len(sub_maps) > 1:
                    # Don't set the block size if there are submaps. (sub_maps contains also the current map)
                    print("🧱  Keeping block size, map has submaps: ", node.map,
                          node.map.gpu_block_size, node.map.params, sub_maps)
                else:
                    print(
                        f"🧱  Changing block size: {node.map}: {node.map.gpu_block_size} -> {[total_size, 1, 1]}")
                    node.map.gpu_block_size = [total_size, 1, 1]
            else:
                print("🧱  Keeping block size: ", node.map, node.map.gpu_block_size)


def change_map_schedule(sdfg: dace.SDFG,
                        new_schedule: dace.dtypes.ScheduleType,
                        label_regex: str,
                        expected_params: List[str] = None):
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                and node.schedule == dace.dtypes.ScheduleType.GPU_Device \
                and len(node.map.params):
            if re.fullmatch(label_regex, node.label):
                if expected_params is None or set(node.map.params) == set(expected_params):
                    print(
                        f"⏲  Changing schedule: {node.map}: {node.schedule} --> {new_schedule}")
                    node.schedule = new_schedule
            else:
                print("⏲  Keeping schedule: ", node.map, node.schedule)
