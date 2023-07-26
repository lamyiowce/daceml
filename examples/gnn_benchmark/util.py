from typing import Dict

from daceml.onnx import register_replacement, TORCH_DTYPE_TO_TYPECLASS
from daceml.onnx.nodes import replacement_entries
from examples.gnn_benchmark.implementations import gcn_implementations, \
    gat_implementations
from examples.gnn_benchmark.implementations.common import SparseLayerBase, \
    SpecialInputType

name_to_impl_class: Dict[str, Dict[str, SparseLayerBase]] = {
    "gcn": {
        "csr": gcn_implementations.GCNConvCSR,
        "csr_adapt": gcn_implementations.GCNConvCSRAdapt,
        "coo": gcn_implementations.GCNConvCOO,
        "coo_adapt": gcn_implementations.GCNConvCOOAdapt,
        "coo_cached": gcn_implementations.GCNConvCOOCached,
        "coo_adapt_cached": gcn_implementations.GCNConvCOOAdaptCached,
        "csc": gcn_implementations.GCNConvCSC,
        "csc_alt": gcn_implementations.GCNConvCSCPropagateFirst,
        "csc_adapt": gcn_implementations.GCNConvCSCAdapt,
        "csc_cached": gcn_implementations.GCNConvCSCCached,
        "csc_adapt_cached": gcn_implementations.GCNConvCSCAdaptCached,
        "ellpack_t": gcn_implementations.GCNConvEllpackTransposed,
        "ellpack": gcn_implementations.GCNConvEllpack,
        "semester_thesis": gcn_implementations.GCNConvSemesterThesis,
        "csr_coo": gcn_implementations.GCNConvCSRCOO,
        "csr_coo_adapt": gcn_implementations.GCNConvCSRCOOAdapt,
        "csc_coo_adapt": gcn_implementations.GCNConvCSCCOOAdapt,
        "csc_coo_adapt_cached": gcn_implementations.GCNConvCSCCOOAdaptCached,
    },
    "gat": {
        "semester_thesis": gat_implementations.GATConvSemesterThesis,
        "csr": gat_implementations.GATConvCSR,
        "coo": gat_implementations.GATConvCOO,
        "coo_stable": gat_implementations.GATConvCOOStable,
        "coo_cached": gat_implementations.GATConvCOOCached,
        "coo_stable_cached": gat_implementations.GATConvCOOStableCached,
        "coo_stable_cached_altspmm": gat_implementations.GATConvCOOStableCachedAltSpmm,
        "csr_stable": gat_implementations.GATConvCSRStable,
        "csr_stable_cached": gat_implementations.GATConvCSRStableCached,
        "csc": gat_implementations.GATConvCSC,
    }
}
name_to_impl_class['gcn_single_layer'] = name_to_impl_class['gcn']
name_to_impl_class['gat_single_layer'] = name_to_impl_class['gat']


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
