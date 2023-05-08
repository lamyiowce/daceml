import datetime
import functools
from collections import OrderedDict
from typing import Sequence, Dict, Optional

import torch
from torch import profiler


def run_with_profile(funcs, func_names, warmup: int = 3,
                     profile_tag: Optional[str] = None):
    for f, name in zip(funcs, func_names):
        for _ in range(warmup):
            f()

        with profiler.profile(activities=[profiler.ProfilerActivity.CPU,
                                          profiler.ProfilerActivity.CUDA],
                              profile_memory=True, with_stack=True) as prof:
            f()

        filename = f"{profile_tag or 'profile'}_{name}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        prof.export_chrome_trace(filename + ".json")
        prof.export_stacks(filename + "_gpu.txt", "self_cuda_time_total")
        prof.export_stacks(filename + "_cpu.txt", "self_cpu_time_total")


def torch_profile(experiment_infos: Dict[str, 'ExperimentInfo'],
                  torch_model: torch.nn.Module,
                  torch_csr_args: Sequence[torch.Tensor],
                  torch_edge_list_args: Sequence[torch.Tensor],
                  args,
                  targets: Optional[torch.Tensor],
                  backward: bool,
                  skip_torch_csr: bool = False,
                  skip_torch_edge_list: bool = False,
                  ):
    experiment_infos = OrderedDict(experiment_infos)

    funcs = []
    func_names = []

    if not skip_torch_csr:
        torch_model.eval()
        funcs.append(lambda: torch_model(*torch_csr_args))
        func_names.append('torch_csr')
    if not skip_torch_edge_list:
        torch_model.eval()
        funcs.append(lambda: torch_model(*torch_edge_list_args))
        func_names.append('torch_edge_list')

    profile_tag = f"{args.model}_{args.data}"

    ### Forward pass.

    def run_with_inputs(model, inputs):
        return model(*inputs)

    for impl_spec, experiment_info in experiment_infos.items():
        model = experiment_info.model_eval
        inputs = experiment_info.data.to_input_list()

        funcs.append(functools.partial(run_with_inputs, model, inputs))
        name = "dace"
        if args.threadblock_dynamic:
            name += "_tb-dynamic"
        if args.no_opt:
            name += "_no_autoopt"
        if args.no_persistent_mem:
            name += "_no_persistent_mem"
        name += f"_{impl_spec}"

        func_names.append(name)

    print(f"---> Profiling the forward pass...")
    with torch.no_grad():
        grad_test = funcs[0]()
        assert grad_test.grad_fn is None
        run_with_profile(funcs, func_names, profile_tag=f"{profile_tag}_fwd")


    ### Backward pass.
    if backward:
        assert targets is not None

        if hasattr(torch_model, 'conv2'):
            criterion = torch.nn.NLLLoss()
        else:
            criterion = lambda pred, targets: torch.sum(pred)

        def backward_fn(model, inputs):
            pred = model(*inputs)
            loss = criterion(pred, targets)
            loss.backward()

        backward_funcs = []
        if not skip_torch_csr:
            torch_model.train()
            backward_funcs.append(
                lambda: backward_fn(torch_model, torch_csr_args))
        if not skip_torch_edge_list:
            torch_model.train()
            backward_funcs.append(
                lambda: backward_fn(torch_model, torch_edge_list_args))

        for experiment_info in experiment_infos.values():
            model = experiment_info.model_train
            inputs = experiment_info.data.to_input_list()
            backward_funcs.append(functools.partial(backward_fn, model, inputs))

        print(f"---> Profiling the BACKWARD pass...")
        run_with_profile(backward_funcs,
                                    func_names=func_names,
                                    profile_tag=f"{profile_tag}_bwd")

