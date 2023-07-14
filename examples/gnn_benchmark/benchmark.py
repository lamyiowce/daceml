import faulthandler
import functools
import pathlib
import socket
import statistics
from typing import Sequence, Dict, Optional, Tuple
from collections import OrderedDict

import torch

from examples.gnn_benchmark.experiment_info import ExperimentInfo


def do_benchmark(experiment_infos: Dict[str, ExperimentInfo],
                 torch_experiments: Sequence[
                     Tuple[str, torch.nn.Module, Sequence[torch.Tensor]]],
                 dace_tag: Optional[str],
                 targets: Optional[torch.Tensor],
                 backward: bool,
                 loss_fn: torch.nn.Module,
                 use_gpu: bool,
                 model_name: str,
                 hidden_size: int,
                 outfile: Optional[pathlib.Path] = None,
                 small: bool = False):
    from examples.gnn_benchmark.performance_measurement import \
        print_time_statistics
    if use_gpu:
        from examples.gnn_benchmark.performance_measurement import \
            measure_performance
    else:
        from examples.gnn_benchmark.performance_measurement import \
            measure_performance_cpu as measure_performance

    experiment_infos = OrderedDict(experiment_infos)

    funcs = []
    func_names = []

    ### Forward pass.

    def run_with_inputs(model, inputs):
        return model(*inputs)

    for torch_name, torch_model, torch_inputs in torch_experiments:
        torch_model.eval()
        funcs.append(
            functools.partial(run_with_inputs, torch_model, torch_inputs))
        func_names.append(torch_name)

    for impl_spec, experiment_info in experiment_infos.items():
        model = experiment_info.model_eval
        inputs = experiment_info.data.to_input_list()

        funcs.append(functools.partial(run_with_inputs, model, inputs))
        func_names.append(dace_tag + '_' + impl_spec)

    print(f"---> Benchmarking the forward pass...")
    with torch.no_grad():
        grad_test = funcs[0]()
        assert grad_test.grad_fn is None
        times = measure_performance(funcs,
                                    func_names=func_names,
                                    warmups=10 if not small else 2,
                                    num_iters=10 if not small else 2,
                                    timing_iters=100 if not small else 3)
    print()
    print(f"\n------ {model_name.upper()} {hidden_size} FORWARD RUNTIME [ms] ------")
    print_time_statistics(times, func_names)
    print()

    if outfile is not None:
        write_stats_to_file(func_names, times, model_name, hidden_size, outfile)

    ### Backward pass.
    if backward:
        assert targets is not None

        def backward_fn(model, inputs):
            pred = model(*inputs)
            loss = loss_fn(pred, targets)
            loss.backward()

        backward_funcs = []

        for torch_name, torch_model, torch_inputs in torch_experiments:
            torch_model.train()
            backward_funcs.append(
                functools.partial(backward_fn, torch_model, torch_inputs))

        for experiment_info in experiment_infos.values():
            model = experiment_info.model_train
            inputs = experiment_info.data.to_input_list()
            backward_funcs.append(functools.partial(backward_fn, model, inputs))

        print(f"---> Benchmarking the BACKWARD pass...")
        times = measure_performance(backward_funcs,
                                    func_names=func_names,
                                    warmups=5 if not small else 2,
                                    num_iters=5 if not small else 2,
                                    timing_iters=20 if not small else 3)
        print()
        print(f"\n------ {model_name.upper()} {hidden_size} BACKWARD RUNTIME [ms] ------")
        print_time_statistics(times, func_names)
        print()

        if outfile is not None:
            path = outfile.with_name(outfile.stem + "-bwd.csv")
            write_stats_to_file(func_names, times, model_name, hidden_size,
                                path)


def write_stats_to_file(func_names, times, model_name, hidden_size,
                        file_path: pathlib.Path):
    add_header = not file_path.exists()
    with open(file_path, 'a') as file:
        if add_header:
            # Write out some system info.
            file.write(
                f"# HOST {socket.gethostname()}, GPU {torch.cuda.get_device_name()}\n")

            headers = [
                'Name', 'Model', 'Size', 'Min', 'Mean',
                'Median',
                'Stdev', 'Max'
            ]
            file.write(','.join(headers) + '\n')
        file.write(
            stats_as_csv_entry(times, func_names, model_name, hidden_size))


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
