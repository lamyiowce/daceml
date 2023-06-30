import numpy as np
import scipy
import torch
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor


def check_equal(expected_pred, pred, name=None, do_assert=True, silent=False, atol=1e-6, rtol=1e-6):
    is_correct = np.allclose(pred, expected_pred, atol=atol, rtol=rtol)
    if not silent or not is_correct:
        print('\n' + name if name else '')
        print('Calculated: \n', pred)
        print('Expected: \n', expected_pred)
        if is_correct:
            print("OK")
    if not is_correct:
        max_err_abs = np.abs(expected_pred - pred).max()
        print("Abs error: ", max_err_abs)
        max_err_rel = max_err_abs / np.abs(expected_pred).max()
        print("Rel error: ",
              max_err_rel)
        if do_assert:
            assert False, f"{name} abs: {max_err_abs}, rel: {max_err_rel}"
        return False, f"{name} abs: {max_err_abs}, rel: {max_err_rel}"
    return True, ""


def check_grads(expected_params, result, atol=1e-6, rtol=1e-6):
    messages = []
    for name, param in expected_params.items():
        expected_grad = get_grad_as_numpy(param)
        if result[name] is not None:
            grad = get_grad_as_numpy(result[name])
            correct, message = check_equal(expected_grad,
                                           grad,
                                           name=name + ' grad', do_assert=False,
                                           atol=atol, rtol=rtol)
            if not correct:
                messages.append(message)
                print(message)
    assert len(messages) == 0, str(messages)


def pred_torch(adj_matrix, layer, random_mask, x):
    # PyG requires that the adj matrix is transposed when using SparseTensor.
    pred, att_weights = layer(x, adj_matrix.t(), return_attention_weights=True)
    loss = torch.sum(pred * random_mask)
    pred.retain_grad()
    loss.backward()
    return att_weights, pred


def setup_data(N, F_in, F_out, heads, seed=42):
    dtype = torch.float32
    negative_slope = 0.2
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    layer = GATConv(F_in, F_out, heads=heads, add_self_loops=False,
                    negative_slope=negative_slope, bias=True)
    x = torch.randn(N, F_in, requires_grad=True)
    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    graph.data = np.ones_like(graph.data, dtype=np.float32)
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A).to(dtype=dtype))
    random_mask = torch.arange(N * F_out * heads).resize(N, heads * F_out).to(dtype) / 10
    return adj_matrix, layer, random_mask, x, graph


def get_grad_as_numpy(array):
    if hasattr(array, 'grad'):
        if array.grad is not None:
            grad = np.array(array.grad.detach().cpu())
        else:
            grad = np.array(array.detach().cpu())
    elif isinstance(array, np.ndarray):
        if hasattr(array, 'asnumpy'):
            grad = array.asnumpy()
        else:
            grad = array
    else:
        grad = array.get()

    return grad
