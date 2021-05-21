import numpy as np
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from daceml.pytorch import DaceModule
from daceml.testing import torch_tensors_close, copy_to_gpu


def run_pytorch_module(module,
                       sdfg_name,
                       gpu,
                       shape=None,
                       use_max=False,
                       auto_optimize=True):
    shape = shape or (3, 5)

    module = copy_to_gpu(gpu, module)

    input_value = torch.rand(*shape, dtype=torch.float32)

    pytorch_input = torch.empty(
        *shape,
        dtype=torch.float32,
        requires_grad=False,
    )
    pytorch_input.copy_(input_value)

    dace_input = torch.empty(*shape, dtype=torch.float32, requires_grad=False)
    dace_input.copy_(input_value)

    dace_input = copy_to_gpu(gpu, dace_input)
    pytorch_input = copy_to_gpu(gpu, pytorch_input)

    pytorch_input.requires_grad = True
    dace_input.requires_grad = True

    if use_max:
        pytorch_s = module(pytorch_input).max()
    else:
        pytorch_s = module(pytorch_input).sum()
    pytorch_s.backward()

    dace_module = DaceModule(module,
                             backward=True,
                             sdfg_name=sdfg_name,
                             auto_optimize=auto_optimize)

    if use_max:
        dace_s = dace_module(dace_input).max()
    else:
        dace_s = dace_module(dace_input).sum()
    dace_s.backward()
    torch_tensors_close("output", pytorch_input.grad, dace_input.grad)


def test_simple(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.log(x)
            return x

    run_pytorch_module(Module(), sdfg_name, gpu)


def test_repeated(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            x = torch.sqrt(x)
            return x

    run_pytorch_module(Module(), sdfg_name, gpu)


def test_softmax(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def forward(self, x):
            x = F.softmax(x, dim=1)
            return x

    run_pytorch_module(Module(), sdfg_name, gpu, use_max=True)


def test_reshape_on_memlet_path(sdfg_name, gpu):
    # required test: this function in a nn.Module, with apply strict so that the reshape is
    # inlined and copy is removed
    class Module(torch.nn.Module):
        def forward(self, x):
            reshaped = torch.reshape(x + 1, [3, 3])
            return torch.log(reshaped) + torch.reshape(
                torch.tensor([[3, 2, 1]], device=reshaped.device), [3])

    run_pytorch_module(Module(), sdfg_name, gpu, shape=(9, ))


def test_weights_ln(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.ln = nn.LayerNorm(32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.ln(x)
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, gpu, shape=(4, 784), use_max=False)


def test_layernorm(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.ln = nn.LayerNorm(3)

        def forward(self, x):
            return self.ln(x)

    run_pytorch_module(Module(), sdfg_name, gpu, shape=(1, 3), use_max=True)


def test_weights(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    run_pytorch_module(Module(), sdfg_name, gpu, shape=(4, 784), use_max=False)


def test_nested_gradient_summation(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.rand(10, 10))

        def forward(self, x):
            y = x @ self.fc1
            z = x * 2
            return z + y

    run_pytorch_module(Module(), sdfg_name, gpu, shape=(4, 10), use_max=False)


def test_trans_add(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()

        def forward(self, x):
            x = x + 1
            x = torch.transpose(x.reshape(4, 4), 1, 0)
            return x

    run_pytorch_module(Module(),
                       sdfg_name,
                       gpu,
                       shape=(16, ),
                       use_max=False,
                       auto_optimize=True)


def test_batched_matmul(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.fc1 = nn.Parameter(torch.ones([10, 5, 3]))

        def forward(self, x):
            return self.fc1 @ x

    run_pytorch_module(Module(), sdfg_name, gpu, use_max=False)


def test_scalar_forwarding(sdfg_name, gpu):
    class Module(torch.nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.factor = nn.Parameter(torch.ones(()))

        def forward(self, x):
            return self.factor * x

    run_pytorch_module(Module(), sdfg_name, gpu, use_max=False)
