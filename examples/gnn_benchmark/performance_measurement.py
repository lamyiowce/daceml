"""Utilities for profiling PyTorch."""

import statistics

import torch
import torch.cuda
import tabulate

# Validate that we have CUDA and initialize it on load.
from daceml.testing.profiling import binary_utils
from daceml.testing.profiling.event_profiler import CudaTimer, \
    _DEFAULT_WAIT_TIME

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('No CUDA support or GPUs available')
torch.cuda.init()

# A device for CUDA.
cuda_device = torch.device('cuda:0')

def measure_performance(funcs,
               name='',
               func_names=None,
               num_iters=10,
               timing_iters=100,
               warmups=5,
               launch_wait=False):
    """ Run and time funcs.

        :param funcs: a list of functions that do GPU work.
        :param name: a name to be used for NVTX ranges.
        :param func_names: a list of names to be used for NVTX ranges for each function.
        :param num_iters: the number of iterations to perform.
        :param warmups:  the number of warmup iterations to perform.
        :param launch_wait: if True, launches a wait kernel before measuring to hide kernel launch latency.
        :return: the time, in ms, of each function in funcs on each iteration.
    """
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

        iter_times = timer.get_times()
        for t in iter_times:
            times[i].append(t / timing_iters)
    binary_utils.stop_cuda_profiling()
    return times