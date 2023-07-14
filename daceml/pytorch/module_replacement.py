from typing import Dict, Tuple, Callable, List

import torch

from daceml.onnx.nodes.replacement import MODULES_TO_REPLACE


def replace_modules(module: torch.nn.Module):
    replaced_idx = 0
    placeholder_id_to_module: Dict[int, Tuple[str, torch.nn.Module]] = {}

    def replace_modules_helper(module: torch.nn.Module, prefix: str):
        nonlocal replaced_idx
        nonlocal placeholder_id_to_module
        for name, submodule in module.named_children():
            cls = submodule.__class__
            cls_name = f"{cls.__module__}.{cls.__qualname__}"
            local_prefix = f'{prefix}{name}.'
            if cls_name in MODULES_TO_REPLACE:
                replacement_info = MODULES_TO_REPLACE[cls_name]
                torch_out_specs = [spec.as_torch_spec(submodule) for spec in
                                   replacement_info.outputs]
                torch_buffer_specs = [spec.as_torch_spec(submodule) for spec in
                                      replacement_info.buffers]
                placeholder = GenericPlaceholder(cls_name, submodule,
                                                 replaced_idx, local_prefix,
                                                 torch_out_specs, torch_buffer_specs)
                setattr(module, name, placeholder)
                placeholder_id_to_module[replaced_idx] = (local_prefix,
                                                          submodule)
                replaced_idx += 1
            else:
                replace_modules_helper(submodule, local_prefix)

    replace_modules_helper(module, prefix='')
    return placeholder_id_to_module


def create_placeholder_function_class(name, module_id,
                                      output_specs: List[Tuple[str, torch.dtype, Callable]]):
    if len(output_specs) == 1:
        @staticmethod
        def forward(ctx, *inputs):
            _, output_dtype, output_shape_fn = output_specs[0]
            return torch.zeros(output_shape_fn(*inputs), dtype=output_dtype)
    else:
        @staticmethod
        def forward(ctx, *inputs):
            return tuple([torch.zeros(output_shape_fn(*inputs), dtype=output_dtype) for
                          _, output_dtype, output_shape_fn in output_specs])

    @staticmethod
    def symbolic(g: torch._C.Graph, *inputs):
        return g.op(f'daceml::{name}', *inputs, module_id_i=module_id, outputs=len(output_specs))

    attrs = {}
    attrs['symbolic'] = symbolic
    attrs['forward'] = forward
    cls = type(name, (torch.autograd.Function,), attrs)
    return cls


class GenericPlaceholder(torch.nn.Module):
    def __init__(self, placeholder_name: str,
                 replaced_module: torch.nn.Module,
                 module_id: int, prefix: str,
                 output_specs: List[Tuple[str, torch.dtype, Callable]],
                 buffer_specs: List[Tuple[str, torch.dtype, Callable]]):
        super().__init__()
        assert len(output_specs) == 1
        self.prefix: str = prefix
        self.placeholder_function = create_placeholder_function_class(
            placeholder_name, module_id, output_specs + buffer_specs)
        for name, p in replaced_module.named_parameters(recurse=False):
            self.register_parameter(name, p)

        for name, dtype, shape_fn in buffer_specs:
            self.register_buffer(name, tensor=None, persistent=False)
        self.buffer_specs = buffer_specs

        for name, submodule in replaced_module.named_modules():
            if len(name) > 0:
                self.add_module(name, submodule)

    def forward(self, *inputs, **kwargs):
        if kwargs:
            raise NotImplementedError("kwargs provided but not supported.")

        if len(self.buffer_specs) == 0:
            output = self.placeholder_function.apply(*inputs)
        else:
            output, *buffers = self.placeholder_function.apply(*inputs)
            for array, (name, _, _) in zip(buffers, self.buffer_specs):
                self._buffers[name] = array
        return output
