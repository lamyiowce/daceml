"""Utilities for profiling PyTorch."""
import statistics
import timeit
from time import sleep

import tabulate
import torch
import torch.cuda


def measure_performance(funcs,
                        func_names=None,
                        num_iters=10,
                        timing_iters=100,
                        warmups=5):
    """ Run and time funcs.

        :param funcs: a list of functions that do GPU work.
        :param name: a name to be used for NVTX ranges.
        :param func_names: a list of names to be used for NVTX ranges for each function.
        :param num_iters: the number of iterations to perform.
        :param warmups:  the number of warmup iterations to perform.
        :param launch_wait: if True, launches a wait kernel before measuring to hide kernel launch latency.
        :return: the time, in ms, of each function in funcs on each iteration.
    """
    from daceml.testing.profiling import binary_utils
    from daceml.testing.profiling.event_profiler import CudaTimer

    times = [list() for _ in range(len(funcs))]
    binary_utils.start_cuda_profiling()
    torch.cuda.init()
    for i, (f, fname) in enumerate(zip(funcs, func_names)):
        torch.cuda.nvtx.range_push(fname + ' Warmup')
        for _ in range(warmups):
            f()
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        timer = CudaTimer(num_iters)

        torch.cuda.nvtx.range_push(fname + ' Iter')
        timer.start()
        for _ in range(num_iters):
            for _ in range(timing_iters):
                f()
            torch._C._cuda_synchronize()
            timer.next()
        timer.end()
        torch.cuda.nvtx.range_pop()

        # Single run for profiling.
        torch.cuda.nvtx.range_push(fname + ' single run.')
        f()
        torch._C._cuda_synchronize()
        torch.cuda.nvtx.range_pop()

        iter_times = timer.get_times()
        for t in iter_times:
            times[i].append(t / timing_iters)
    binary_utils.stop_cuda_profiling()
    return times


def measure_performance_cpu(funcs,
                            func_names=None,
                            num_iters=10,
                            timing_iters=100,
                            warmups=5):
    """ Run and time funcs on a CPU."""
    funcs += [lambda: sleep(0.001)]
    func_names += ['sleep0.001']
    times = [list() for _ in range(len(funcs))]
    for i, (f, fname) in enumerate(zip(funcs, func_names)):
        for _ in range(warmups):
            f()
        timer = timeit.Timer(f)
        measurements = timer.repeat(repeat=num_iters, number=timing_iters)

        times[i] = [1000 * t / timing_iters for t in measurements]
    return times


def print_time_statistics(times, func_names):
    """ Print timing statistics.
    :param times: the result of time_funcs.
    :param func_names: a name to use for each function timed.
    """
    headers = ['Name', 'Min', 'Mean', 'Median', 'Stdev', 'Max']
    rows = []
    for name, func_time in zip(func_names, times):
        rows.append([
            name,
            min(func_time),
            statistics.mean(func_time),
            statistics.median(func_time),
            statistics.stdev(func_time) if len(func_time) > 1 else 0.0,
            max(func_time)
        ])
    print(
        tabulate.tabulate(rows,
                          headers=headers,
                          floatfmt='.4f',
                          tablefmt='github'))
