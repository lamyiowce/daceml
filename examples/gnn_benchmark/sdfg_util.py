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


def change_storage(sdfg: dace.SDFG, array_name: str, new_storage=dace.StorageType.Register,
                   expected_shape: Tuple[int, ...] = None):
    def set_storage(sdfg):
        for state in sdfg.nodes():
            for dnode in state.data_nodes():
                if 'reserved_' in dnode.data:
                    continue
                arr = sdfg.arrays[dnode.data]
                if (arr.transient and not isinstance(arr, dace.data.View)
                        and arr.storage != new_storage):
                    if expected_shape is None or arr.shape == expected_shape:
                        if re.fullmatch(array_name, dnode.data):
                            print(f"  Setting storage for {dnode} to {new_storage} from {arr.storage}.")
                            arr.storage = new_storage
                            if arr.lifetime == dace.dtypes.AllocationLifetime.Persistent:
                                print(
                                    f"  Setting lifetime for {dnode} to Scope from {arr.lifetime}.")
                                arr.lifetime = dace.AllocationLifetime.Scope

    for sub_sdfg in sdfg.all_sdfgs_recursive():
        set_storage(sub_sdfg)


def ensure_datatype(sdfg: dace.SDFG, dtype: dace.dtypes.typeclass, array_name: str = None,
                    expected_shape: Tuple[int, ...] = None):
    def set_datatype(sdfg):
        for state in sdfg.nodes():
            for dnode in state.data_nodes():
                if 'reserved_' in dnode.data:
                    continue
                arr = sdfg.arrays[dnode.data]
                if (arr.transient and not isinstance(arr, dace.data.View)
                        and arr.dtype == dace.float64):
                    if expected_shape is None or arr.shape == expected_shape:
                        if array_name is None or re.fullmatch(array_name, dnode.data):
                            print(f"  Setting dtype for {dnode} to {dtype} from {arr.dtype}.")
                            arr.dtype = dtype

    for sub_sdfg in sdfg.all_sdfgs_recursive():
        set_datatype(sub_sdfg)


def apply_to_both(fn: Callable[[dace.SDFG], None]):
    def wrapper(forward_sdfg, backward_sdfg):
        fn(forward_sdfg)
        fn(backward_sdfg)

    return wrapper


def set_library_node_implementation(sdfg: dace.SDFG,
                                    implementation_name: str,
                                    node_name: str = None,
                                    node_class: type = None,
                                    schedule: dace.dtypes.ScheduleType = None,
                                    verbose=False):
    def matches(node: nd.LibraryNode):
        if node_name is not None and node.label != node_name:
            return False
        if node_class is not None and not isinstance(node, node_class):
            return False
        return True

    counter = 0
    counter_already = 0

    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nd.LibraryNode):
            if matches(node):
                if node.implementation == implementation_name:
                    counter_already += 1
                else:
                    print(
                        f"📚  Setting impl {node} ({node.schedule}) from {node.implementation} to {implementation_name}.")
                    node.implementation = implementation_name
                    node.schedule = schedule or node.schedule
                    counter += 1
            elif verbose:
                print(f"📚  Skipping node {node}.")

    print(
        f"📚 📚 📚  Set {counter} nodes of {node_name} / {node_class}  to {implementation_name}, {counter_already} were already set.")


def set_reduce_implementation(sdfg: dace.SDFG, implementation_name: str = 'GPUAuto'):
    set_library_node_implementation(sdfg, implementation_name,
                                    node_class=dace.libraries.standard.nodes.Reduce)


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
                (node.map,
                 {dace.symbol(k): dace.symbol(k) for k in node.map.range.free_symbols}))
    return res


def flatten_blocks_for_1d_maps(sdfg: dace.SDFG, verbose=False):
    default_block_size = [int(b) for b in
                          Config.get('compiler', 'cuda', 'default_block_size').split(',')]
    total_size = functools.reduce(lambda a, b: a * b, default_block_size)

    for node, dfg_scope in sdfg.all_nodes_recursive():
        if isinstance(node, nd.MapEntry) \
                and node.schedule == dace.dtypes.ScheduleType.GPU_Device \
                and len(node.map.params):
            subgraph = dfg_scope.scope_subgraph(node)
            sub_maps = get_tb_maps_recursive(subgraph)
            if len(sub_maps) == 1:
                # If map has only one param, change to [max, 1, 1]
                if len(node.map.params) == 1 and node.map.gpu_block_size is None:
                    print(
                        f"🧱  Changing block size: {node.map}: {node.map.gpu_block_size} -> {[total_size, 1, 1]}")
                    node.map.gpu_block_size = [total_size, 1, 1]
                else:
                    new_block_sizes = adjust_block_size(default_block_size, node, total_size)

                    if new_block_sizes != default_block_size:
                        print(
                            f"🧱  Changing block size: {node.map}: {node.map.gpu_block_size} -> {new_block_sizes}")
                        node.map.gpu_block_size = new_block_sizes
                    elif verbose:
                        print(
                            f"🧱  Keeping block size: {node.map}: {node.map.gpu_block_size}.")
            elif verbose:
                # Don't set the block size if there are submaps. (sub_maps contains also the current map)
                print("🧱  Keeping block size, map has submaps: ", node.map,
                      node.map.gpu_block_size, node.map.params, sub_maps)
            elif verbose:
                print("🧱  Keeping block size: ", node.map, node.map.gpu_block_size)


def adjust_block_size(default_block_size, node, total_size):
    new_block_sizes = []
    remaining_size = total_size
    map_sizes = []
    for map_range, block_dim in zip(reversed(node.map.range), default_block_size):
        start, end, step = map_range
        range_len = (end - start + 1) // step
        map_sizes.append(range_len)
        if block_dim > range_len:
            new_block_sizes.append(max(1, range_len))
            remaining_size = max(1, remaining_size // range_len)
        else:
            new_block_sizes.append(-1)
    new_block_sizes = new_block_sizes + [1] * 3
    new_block_sizes = new_block_sizes[:3]
    print(node)
    print("Map range, Block sizes:", map_sizes, new_block_sizes)
    num_free = new_block_sizes.count(-1)

    if len(map_sizes) == 1:
        # If the map has only one dim, set it to remaining size
        new_block_sizes = [total_size, 1, 1]
    if num_free == 1:
        # Block sizes of form [64, 8, -1] or [-1, 8, 8] or [40, -1, 8]
        new_block_sizes[new_block_sizes.index(-1)] = remaining_size
    elif num_free == 2:
        if len(map_sizes) == 2:
            # Both dims bigger than block size, set to default.
            # We assume the default block size is [64, 8, 1]
            new_block_sizes = default_block_size
        else:
            # Block sizes of form [64, -1, -1] or [-1, -1, 8] or [-1, 7, -1]
            # Set the first dim to remaining, last to 1
            new_block_sizes[new_block_sizes.index(-1)] = remaining_size
            new_block_sizes[new_block_sizes.index(-1)] = 1
    elif num_free == 3:
        # [-1, -1, -1], just use the default (array is big enough).
        new_block_sizes = default_block_size
    print("New block sizes:", new_block_sizes)
    return new_block_sizes


def change_map_schedule(sdfg: dace.SDFG,
                        new_schedule: dace.dtypes.ScheduleType,
                        label_regex: str,
                        expected_params: List[str] = None,
                        verbose=False):
    print(
        f"⏲ ⏲ ⏲ Changing maps fitting {label_regex} schedules (params {expected_params}) to {new_schedule}")
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                and len(node.map.params):
            if re.fullmatch(label_regex, node.label):
                if expected_params is None or set(node.map.params) == set(expected_params):
                    print(
                        f"⏲  Changing schedule: {node.map}: {node.schedule} --> {new_schedule}")
                    node.schedule = new_schedule
                elif verbose:
                    print(
                        f"⏲  Keeping schedule: {node.map}: {node.schedule} (expected params: {expected_params})")
            elif verbose:
                print("⏲  Keeping schedule: ", node.map, node.schedule)
