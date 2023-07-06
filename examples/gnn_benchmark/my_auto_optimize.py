from typing import Union, Dict

import sympy
from dace import config, dtypes
from dace.sdfg import SDFG, nodes, graph as gr
from dace.sdfg import infer_types
from dace.sdfg.state import SDFGState
from dace.transformation import helpers as xfh
from dace.transformation.auto import fpga as fpga_auto_opt
from dace.transformation.auto.auto_optimize import set_fast_implementations, greedy_fuse, tile_wcrs, \
    move_small_arrays_to_stack, make_transients_persistent
from dace.transformation.dataflow import MapCollapse, TrivialMapElimination
from dace.transformation.interstate import LoopToMap, RefineNestedAccess

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def my_auto_optimize(sdfg: SDFG,
                     device: dtypes.DeviceType,
                     validate: bool = True,
                     validate_all: bool = False,
                     symbols: Dict[str, int] = None) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:

        * Simplify
        * Auto-parallelization (loop-to-map)
        * Greedy application of SubgraphFusion
        * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
        * Tiled stream accumulation (MapTiling -> AccumulateTransient)
        * Collapse all maps to parallelize across all dimensions
        * Set all library nodes to expand to ``fast`` expansion, which calls
          the fastest library on the target device

    :param sdfg: The SDFG to optimize.
    :param device: the device to optimize for.
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :param symbols: Optional dict that maps symbols (str/symbolic) to int/float
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """
    debugprint = config.Config.get_bool('debugprint')

    # Simplification and loop parallelization
    transformed = True
    sdfg.apply_transformations_repeated(TrivialMapElimination, validate=validate,
                                        validate_all=validate_all)
    while transformed:
        sdfg.simplify(validate=False, validate_all=validate_all)
        for s in sdfg.sdfg_list:
            xfh.split_interstate_edges(s)
        l2ms = sdfg.apply_transformations_repeated((LoopToMap, RefineNestedAccess),
                                                   validate=False,
                                                   validate_all=validate_all)
        transformed = l2ms > 0

    # Collapse maps and eliminate trivial dimensions
    sdfg.simplify()
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)

    # Apply GPU transformations and set library node implementations

    if device == dtypes.DeviceType.GPU:
        sdfg.apply_gpu_transformations()
        sdfg.simplify()

    #### EDIT
    from examples.gnn_benchmark.sdfg_util import change_map_schedule, \
        set_library_node_implementation

    if device == dtypes.DeviceType.GPU:
        change_map_schedule(sdfg,
                            new_schedule=dtypes.ScheduleType.Sequential,
                            label_regex=r'examples_gnn_benchmark_implementations_gat_implementations_torch_geometricDOTnnDOTconvDOTgat_convDOTGATConv_0_expansion_\d+',
                            expected_params=['h'])
        change_map_schedule(sdfg,
                            new_schedule=dtypes.ScheduleType.Sequential,
                            label_regex=r'examples_gnn_benchmark_implementations_gat_implementations_gat_op_\d+',
                            expected_params=['h'])
        change_map_schedule(sdfg,
                            new_schedule=dtypes.ScheduleType.Sequential,
                            label_regex=r'call_\d+_map',
                            expected_params=['h'])
        change_map_schedule(sdfg,
                            new_schedule=dtypes.ScheduleType.Sequential,
                            label_regex=r'examples_gnn_benchmark_implementations_gat_backward_backward_fn_\d+',
                            expected_params=['h'])
        change_map_schedule(sdfg,
                            new_schedule=dtypes.ScheduleType.Sequential,
                            label_regex=r'examples_gnn_benchmark_implementations_gat_backward_basic_gat_backward_\d+',
                            expected_params=['h'])

    #### END EDIT

    # fuse subgraphs greedily
    sdfg.simplify()

    greedy_fuse(sdfg, device=device, validate_all=validate_all)

    # fuse stencils greedily
    greedy_fuse(sdfg, device=device, validate_all=validate_all, recursive=False, stencil=True)

    # Move Loops inside Maps when possible
    from dace.transformation.interstate import MoveLoopIntoMap
    sdfg.apply_transformations_repeated([MoveLoopIntoMap])

    if device == dtypes.DeviceType.FPGA:
        # apply FPGA Transformations
        sdfg.apply_fpga_transformations()
        fpga_auto_opt.fpga_global_to_local(sdfg)
        fpga_auto_opt.fpga_rr_interleave_containers_to_banks(sdfg)

        # Set all library nodes to expand to fast library calls
        set_fast_implementations(sdfg, device)
        return sdfg

    # Tiled WCR and streams`
    for nsdfg in list(sdfg.all_sdfgs_recursive()):
        tile_wcrs(nsdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse, validate=False, validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        # Set OMP collapse property to map length
        if isinstance(node, nodes.MapEntry):
            # FORNOW: Leave out
            # node.map.collapse = len(node.map.range)
            pass

    # Set all library nodes to expand to fast library calls
    set_fast_implementations(sdfg, device)

    ########## EDIT
    if device == dtypes.DeviceType.GPU:
        set_library_node_implementation(sdfg, implementation_name='cuBLAS',
                                        node_name='_MatMult_gemv',
                                        schedule=dtypes.ScheduleType.GPU_Device)
        set_library_node_implementation(sdfg, implementation_name='cuSPARSE',
            node_name='coomm',
            schedule=dtypes.ScheduleType.GPU_Device)

        set_library_node_implementation(sdfg, implementation_name='cuSPARSE',
            node_name='csrmm',
            schedule=dtypes.ScheduleType.GPU_Device)
    # ########## END EDIT

    # NOTE: We need to `infer_types` in case a LibraryNode expands to other LibraryNodes (e.g., np.linalg.solve)
    infer_types.infer_connector_types(sdfg)
    infer_types.set_default_schedule_and_storage_types(sdfg, None)
    sdfg.expand_library_nodes()

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections on a per-SDFG basis
    for nsdfg in sdfg.all_sdfgs_recursive():
        nsdfg.openmp_sections = False

    # Set all Default storage types that are constant sized to registers
    move_small_arrays_to_stack(sdfg)

    # Make all independent arrays persistent
    make_transients_persistent(sdfg, device)

    if symbols:
        # Specialize for all known symbols
        known_symbols = {s: v for (s, v) in symbols.items() if s in sdfg.free_symbols}
        known_symbols = {}
        for (s, v) in symbols.items():
            if s in sdfg.free_symbols:
                if isinstance(v, (int, float)):
                    known_symbols[s] = v
                if isinstance(v, sympy.core.numbers.Integer):
                    try:
                        known_symbols[s] = int(v)
                    except TypeError:
                        pass

        if debugprint and len(known_symbols) > 0:
            print("Specializing the SDFG for symbols", known_symbols)
        sdfg.specialize(known_symbols)

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
