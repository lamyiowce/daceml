import dace
import numpy as np
import pytest
import scipy
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from daceml.torch.module import DaceModule
from examples.gnn_benchmark.csrmm_libnode import csrmm
from examples.gnn_benchmark.implementations.gat_implementations import \
    GATConvCSR
from examples.gnn_benchmark.util import register_replacement_overrides

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def set_implementation(dace_module, implementation):
    def hook(dace_module):
        for node, _ in dace_module.sdfg.all_nodes_recursive():
            if (isinstance(node, dace.sdfg.nodes.LibraryNode)
                    and implementation in node.implementations):
                node.implementation = implementation

    dace_module.prepend_post_onnx_hook("set_implementation", hook)


@pytest.mark.parametrize("bias", [True], ids=['bias'])
@pytest.mark.parametrize("implementation", ['csr'])
@pytest.mark.parametrize("N,F,heads", [(2, 3, 1)])
# @pytest.mark.parametrize("N,F,heads", [(2, 3, 1), (2, 3, 2), (120, 20, 8)])
@pytest.mark.parametrize("seed", [40])
def test_gat(bias, implementation, N, F, heads, seed):
    F_in = F
    F_out = F + 3
    torch.random.manual_seed(42)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    register_replacement_overrides(implementation_name=implementation,
                                   layer_name='gat', idx_dtype=torch.int64,
                                   val_dtype=torch.float32)

    sdfg_name = f'GAT_{implementation}_{bias}_{heads}_{seed}'

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(F_in,
                                 F_out,
                                 bias=bias,
                                 heads=heads,
                                 add_self_loops=False)

        def forward(self, x, *edge_info):
            x = self.conv1(x, *edge_info)
            return x

    reference_model = GAT().to(device)
    model = DaceModule(reference_model, sdfg_name=sdfg_name)
    set_implementation(model, implementation)

    # Create input.
    graph = scipy.sparse.random(N, N, density=0.5,
                                format='csr') + scipy.sparse.eye(N,
                                                                 format='csr')
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A).to(device))
    adj_matrix.set_value(None)
    print(adj_matrix)
    rowptr, col, _ = adj_matrix.csr()
    rowptr = rowptr.to(device)
    col = col.to(device)
    x = torch.rand((N, F_in)).to(device)

    # PyG requires that the adj matrix is transposed when using SparseTensor.
    expected_pred = reference_model(x, adj_matrix.t()).cpu().detach().numpy()

    pred = model(x, rowptr, col).cpu().detach().numpy()

    # my_gat_op = GATConvCSR.make_op(N=N, heads=heads, num_out_features=F_out,
    #                                num_entries=col.shape[0],
    #                                dtype=np.float32,
    #                                negative_slope=0.2, do_bias=True)
    #
    # plain_pred = torch.zeros((N, F_out), dtype=torch.float32)
    # my_gat_op(x.detach(), rowptr.detach(), col.detach(),
    #           lin_srcDOTweight=reference_model.conv1.lin_src.weight.detach(),
    #           att_src=reference_model.conv1.att_src.detach(),
    #           att_dst=reference_model.conv1.att_dst.detach(),
    #           bias=reference_model.conv1.bias.detach(),
    #           output=plain_pred)

    # check_equal(plain_pred, pred)
    check_equal(expected_pred, pred)


def check_equal(expected_pred, pred, name=None):
    print('\n' + name)
    print('Calculated: \n', pred)
    print('Expected: \n', expected_pred)
    if not np.allclose(pred, expected_pred):
        max_err_abs = np.abs(pred - expected_pred).max()
        print("Abs error: ", max_err_abs)
        max_err_rel = max_err_abs / np.abs(expected_pred).max()
        print("Rel error: ",
              max_err_rel)
        assert False, f"{name} abs: {max_err_abs}, rel: {max_err_rel}"


def test_gat_compare_gpu():
    import cupy as cp
    N = 2
    heads = 3
    F_in = 3
    F_out = 4
    dtype = np.float32
    negative_slope = 0.2

    torch.random.manual_seed(42)
    np.random.seed(42)
    cp.random.seed(42)

    # Create input.
    graph = (scipy.sparse.random(N, N, density=0.5, format='csr')
             + scipy.sparse.eye(N, format='csr'))
    graph.data = cp.ones_like(graph.data)
    rowptr, col = np.ascontiguousarray(graph.indptr), np.ascontiguousarray(
        graph.indices)
    rowptr = cp.asarray(rowptr)
    col = cp.asarray(col)
    num_entries = col.shape[0]
    x = cp.random.rand(N, F_in).astype(dtype)
    output = cp.zeros((N, heads * F_out), dtype=dtype)
    out_e_before_softmax = cp.zeros((num_entries, heads), dtype=dtype)
    out_e_softmax_sum = cp.zeros((N, heads), dtype=dtype)
    out_e = cp.zeros((num_entries, heads), dtype=dtype)
    out_alpha_src = cp.zeros((N, heads), dtype=dtype)
    out_alpha_dst = cp.zeros((N, heads), dtype=dtype)

    layer = GATConv(F_in, F_out, bias=True, heads=heads, add_self_loops=False)

    @dace.program
    def gat(node_features, rowptrs, columns, lin_srcDOTweight, att_src, att_dst,
            output, e, e_before_softmax, e_softmax_sum, alpha_src, alpha_dst):
        # features = np.zeros((N, heads, F_out),
        #                     dtype=dtype)
        # features_tmp = np.einsum('ij,kj->ik', node_features,
        #                          lin_srcDOTweight)
        #
        # # features: N x H x F'
        # features[:] = np.reshape(features_tmp,
        #                          (N, heads, F_out))

        # Compute node attention coefficients.
        alpha_src[:] = np.sum(node_features * att_src, axis=-1)  # shape: N x H
        alpha_dst[:] = np.sum(node_features * att_dst, axis=-1)  # N x H

        # Calculate attention weights.
        e_softmax_sum[:] = 0

        # TODO: Below loop can be flipped.
        for l in dace.map[0:N]:
            for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                # Calculating e_l->colv
                colv = columns[v]
                e_tmp = alpha_src[l] + alpha_dst[colv]
                # LeakyReLU
                e_tmp = np.maximum(negative_slope * e_tmp, e_tmp)
                e_tmp = np.exp(e_tmp)
                e_before_softmax[v] = e_tmp
                e_softmax_sum[colv] += e_tmp

        # # Softmax normalization.
        # for j in dace.map[0:num_entries]:
        #     colj = columns[j]
        #     e[j] = e_before_softmax[j] / e_softmax_sum[colj]
        #
        # if heads == 1:
        #     csrmm(rowptrs, columns, e,
        #           np.reshape(features, (N, F_out)),
        #           output,
        #           transA=True, beta=0.0)
        # else:
        #     output_perm = np.zeros((heads, N, F_out),
        #                            dtype=dtype)  # H x N x F'
        #     features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'
        #     e_perm = np.transpose(e, (1, 0))  # H x num_entries
        #     # for h in dace.map[0:heads]@dace.dtypes.ScheduleType.Unrolled:
        #     for h in range(heads):
        #         csrmm(rowptrs, columns, e_perm[h], features_perm[h],
        #               output_perm[h],
        #               transA=True,
        #               beta=1.0)
        #
        #     output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
        #                            (N, heads * F_out))

    weight = cp.array(layer.lin_src.weight.detach().numpy())
    att_src = cp.array(layer.att_src.detach().numpy())
    att_dst = cp.array(layer.att_dst.detach().numpy())
    sdfg = gat.to_sdfg(x, rowptr, col, weight, att_src,
                       att_dst, output, out_e, out_e_before_softmax,
                       out_e_softmax_sum,
                       out_alpha_src, out_alpha_dst)
    sdfg.apply_gpu_transformations()

    sdfg(node_features=x, rowptrs=rowptr, columns=col, lin_srcDOTweight=weight,
         att_src=att_src, att_dst=att_dst, output=output, e=out_e,
         e_before_softmax=out_e_before_softmax, e_softmax_sum=out_e_softmax_sum,
         alpha_src=out_alpha_src, alpha_dst=out_alpha_dst)

    expected = np.zeros((N, heads * F_out), dtype=dtype)
    expected_e = np.zeros((num_entries, heads), dtype=dtype)
    expected_e_before_softmax = np.zeros((num_entries, heads), dtype=dtype)
    expected_e_softmax_sum = np.zeros((N, heads), dtype=dtype)
    expected_alpha_src = np.zeros((N, heads), dtype=dtype)
    expected_alpha_dst = np.zeros((N, heads), dtype=dtype)
    gat.f(node_features=x.get(), rowptrs=rowptr.get(), columns=col.get(),
          lin_srcDOTweight=weight.get(),
          att_src=att_src.get(), att_dst=att_dst.get(), output=expected,
          e=expected_e, e_before_softmax=expected_e_before_softmax,
            e_softmax_sum=expected_e_softmax_sum,
          alpha_src=expected_alpha_src, alpha_dst=expected_alpha_dst)

    check_equal(expected_alpha_src, out_alpha_src.get(), 'alpha_src')
    check_equal(expected_alpha_dst, out_alpha_dst.get(), 'alpha_dst')
    check_equal(expected_e_before_softmax, out_e_before_softmax.get(),
                'e_before_softmax')
    check_equal(expected_e_softmax_sum, out_e_softmax_sum.get(),
                'e_softmax_sum')
    check_equal(expected_e, out_e.get(), 'attention_weights')
    check_equal(expected, output.get(), 'result')


@pytest.mark.parametrize("N", [3, 7])
@pytest.mark.parametrize("F", [4, 9])
@pytest.mark.parametrize("heads", [1])
@pytest.mark.parametrize("seed", [2137, 402])
# @pytest.mark.parametrize("N", [3])
# @pytest.mark.parametrize("F", [4])
# @pytest.mark.parametrize("heads", [1])
# @pytest.mark.parametrize("seed", [2137])
def test_spmm_single_head(N, F, heads, seed):
    if torch.cuda.is_available():
        import cupy as np
        def ref(rowptrs, columns, features, e, output):
            output[:] = 0
            for l in range(N):
                for v in range(np.asnumpy(rowptrs[l]),
                               np.asnumpy(rowptrs[l + 1])):
                    colv = columns[v]
                    if heads == 1:
                        for k in dace.map[0:F]:
                            output[colv, k] += e[v, 0] * features[l, 0, k]
    else:
        import numpy as np
        def ref(rowptrs, columns, features, e, output):
            output[:] = 0
            for l in range(N):
                for v in range(rowptrs[l], rowptrs[l + 1]):
                    colv = columns[v]
                    if heads == 1:
                        for k in dace.map[0:F]:
                            output[colv, k] += e[v, 0] * features[l, 0, k]
    torch.random.manual_seed(42)
    np.random.seed(seed)

    @dace.program
    def csrmm_prog(rowptrs, columns, features, e, output):
        csrmm(rowptrs, columns, e, features, output,
              transA=True, beta=0.0)

    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A))
    print(adj_matrix)
    rowptr, col, _ = adj_matrix.csr()
    if torch.cuda.is_available():
        rowptr = np.array(rowptr.numpy(), copy=True)
        col = np.array(col.numpy(), copy=True)
    else:
        rowptr = np.copy(rowptr.numpy())
        col = np.copy(col.numpy())
    x = np.random.rand(N, heads, F)
    vals = np.random.rand(col.shape[0], heads)

    print(f"rowptr {rowptr.shape} {rowptr.dtype}: ", rowptr)
    print(f"col {col.shape}: ", col)
    print(f"x {x.shape}: ", x)
    print(f"vals {vals.shape}: ", vals)

    expected = np.zeros((N, heads * F))
    ref(rowptr, col, x, vals, expected)

    result = np.zeros((N, heads * F))
    csrmm_prog(rowptr, col, x, vals, result)

    check_equal(expected, result)


@pytest.mark.parametrize("N,F", [(3, 4), (7, 9)])
@pytest.mark.parametrize("heads", [2, 3])
@pytest.mark.parametrize("seed", [2137, 402])
# @pytest.mark.parametrize("N,F", [(3, 4)])
# @pytest.mark.parametrize("heads", [2])
# @pytest.mark.parametrize("seed", [2137])
def test_spmm_many_heads_cpu(N, F, heads, seed):
    def ref(rowptrs, columns, features, e, output):
        output[:] = 0
        for l in dace.map[0:N]:
            for v in dace.map[rowptrs[l]:rowptrs[l + 1]]:
                colv = columns[v]
                output[colv] += np.reshape(
                    np.reshape(e[v], (heads, 1)) * features[l],
                    (heads * F,))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.random.manual_seed(42)
    np.random.seed(seed)
    dtype = np.float32

    @dace.program
    def csrmm_prog(rowptrs, columns, features, e, output):
        output_perm = np.zeros((heads, N, F), dtype=dace.float32)  # H x N x F'
        features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'
        e_perm = np.transpose(e, (1, 0))  # H x num_entries
        # for h in dace.map[0:heads]:
        for h in range(heads):
            csrmm(rowptrs, columns, e_perm[h], features_perm[h], output_perm[h],
                  transA=True,
                  beta=1.0)

        output[:] = np.reshape(np.transpose(output_perm, (1, 0, 2)),
                               (N, heads * F))

    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A))
    print(adj_matrix)
    rowptr, col, _ = adj_matrix.csr()
    if torch.cuda.is_available():
        rowptr = np.array(rowptr.numpy(), copy=True)
        col = np.array(col.numpy(), copy=True)
    else:
        rowptr = np.copy(rowptr.numpy())
        col = np.copy(col.numpy())
    x = np.random.rand(N, heads, F).astype(dtype)
    vals = np.random.rand(col.shape[0], heads)

    expected = np.zeros((N, heads * F), dtype=dtype)
    ref(rowptr, col, x, vals, expected)

    # fn_result = torch.zeros((N, heads * F))
    # csrmm_prog.f(rowptr, col, x, vals, fn_result)
    # check_equal(expected, fn_result)

    result = np.zeros((N, heads * F), dtype=dtype)
    csrmm_prog(rowptr, col, x, vals, result)
    check_equal(expected, result)


# @pytest.mark.parametrize("N,F", [(3, 4)])
# @pytest.mark.parametrize("heads", [2])
# @pytest.mark.parametrize("seed", [2137])
@pytest.mark.parametrize("N,F", [(3, 4), (7, 9)])
@pytest.mark.parametrize("heads", [2, 3])
@pytest.mark.parametrize("seed", [2137, 402])
@pytest.mark.gpu
def test_spmm_many_heads_gpu(N, F, heads, seed):
    assert torch.cuda.is_available()
    import cupy as cp

    def ref(rowptrs, columns, features, e, output):
        output[:] = 0
        for l in dace.map[0:N]:
            for v in range(cp.asnumpy(rowptrs[l]), cp.asnumpy(rowptrs[l + 1])):
                colv = columns[v]
                output[colv] += cp.reshape(
                    cp.reshape(e[v], (heads, 1)) * features[l],
                    (heads * F,))

    device = torch.device('cuda')
    torch.random.manual_seed(42)
    np.random.seed(seed)
    dtype = np.float32

    @dace.program
    def csrmm_prog(rowptrs, columns, features, e, output):
        output_perm = np.zeros((heads, N, F), dtype=dace.float32)  # H x N x F'
        # output_perm = np.reshape(np.zeros_like(output), (heads, N, F))  # H x N x F'
        features_perm = np.transpose(features, (1, 0, 2))  # H x N x F'
        e_perm = np.transpose(e, (1, 0))  # H x num_entries
        # for h in dace.map[0:heads]:
        for h in range(heads):
            # e_tmp = np.zeros_like(e_perm[h])
            # e_tmp[:] = e_perm[h]
            # features_tmp = np.zeros_like(features_perm[h])
            # features_tmp[:] = features_perm[h]
            # output_tmp = np.zeros_like(output_perm[h])
            csrmm(rowptrs, columns, e_perm[h], features_perm[h], output_perm[h],
                  transA=True,
                  beta=0.0)
            # output_perm[h, :, :] = np.copy(output_tmp)

            # e_tmp = np.zeros_like(e_perm[h])
            # e_tmp[:] = e_perm[h]
            # features_tmp = np.zeros_like(features_perm[h])
            # features_tmp[:] = features_perm[h]
            # output_tmp = np.zeros_like(output_perm[h])
            # csrmm(rowptrs, columns, e_tmp, features_tmp, output_tmp, transA=True,
            #       beta=0.0)
            # output_perm[h, :, :] = np.copy(output_tmp)

        output[:] = np.copy(
            np.reshape(np.transpose(output_perm, (1, 0, 2)), (N, heads * F)))

    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A))
    print(adj_matrix)
    rowptr, col, _ = adj_matrix.csr()
    rowptr = cp.array(rowptr.numpy(), copy=True)
    col = cp.array(col.numpy(), copy=True)
    x = cp.random.rand(N, heads, F).astype(dtype)
    vals = cp.random.rand(col.shape[0], heads).astype(dtype)

    expected = cp.zeros((N, heads * F), dtype=dtype)
    ref(rowptr, col, x, vals, expected)

    result = cp.zeros((N, heads * F), dtype=dtype)
    sdfg = csrmm_prog.to_sdfg(rowptr, col, x, vals, result)
    sdfg.apply_gpu_transformations()
    sdfg(rowptr, col, x, vals, result)
    check_equal(expected, result)


def test_bwd_weight_compute():
    N = 3
    F_in = 2
    F_out = 4
    # heads = 2
    dtype = torch.float32
    negative_slope = 0.2
    torch.random.manual_seed(42)
    np.random.seed(42)

    def compute_grads(x, adj_matrix, weight,
                      att_src, att_dst,
                      bias,
                      att_weights,
                      output,
                      out_grad):
        # x: N x F_in
        # adj_matrix: N x N
        # weight: F_out x F_in
        # att_src: 1 x H x F_out
        # att_dst: 1 x H x F_out
        # bias: F_out
        # output: N x F_out
        # out_grad: N x F_out

        grads = {'x': None, 'lin_src.weight': None, 'bias': None,
                 'att_src': None,
                 'att_dst': None}
        print(out_grad)
        grads['lin_src.weight'] = (
                x.t() @ att_weights[:, :, 0].t() @ out_grad).t()

        H_prime = x @ weight.t()  # N x F_out
        # H_prime @ att_src =
        alpha_src = torch.sum(H_prime * att_src[0], dim=-1)  # N x H
        alpha_dst = torch.sum(H_prime * att_dst[0], dim=-1)  # N x H
        C = alpha_src[None, :] + alpha_dst[:, None]  # N x N x H
        Tau = adj_matrix.t() * torch.exp(torch.maximum(negative_slope * C, C))
        Tau_sum = torch.sum(Tau, dim=1)[:, None]  # N x 1 x H
        Phi = Tau / Tau_sum  # N x N x H
        assert torch.allclose(Phi[:, :, None], att_weights)

        def d_leaky_relu(x):
            return torch.where(x > 0, torch.ones_like(x),
                               torch.ones_like(x) * negative_slope)

        F_first_denominator = Tau_sum * (out_grad @ H_prime.t())
        # TODO: maybe we need to do [:, None] here.
        F_second_denominator = Tau_sum ** 2 * torch.sum(
            (Tau @ H_prime) * out_grad, dim=1)
        # F: N x N
        F = (d_leaky_relu(C) / F_first_denominator +
             d_leaky_relu(C) / F_second_denominator)

        # Gamma: N x F_in
        Gamma = x * (torch.sum(F, dim=1) * alpha_src +
                     torch.sum(F, dim=0) * alpha_dst)[:, None]

        # x_grad: N x F_in
        grads['x'] = x * Gamma

        grads['bias'] = torch.sum(out_grad, dim=0)
        return grads

    layer = GATConv(F_in, F_out, heads=1, add_self_loops=False,
                    negative_slope=negative_slope)

    x = torch.randn(N, F_in, requires_grad=True)
    graph = scipy.sparse.random(N, N, density=0.5, format='csr')
    graph.data = np.ones_like(graph.data)
    adj_matrix = SparseTensor.from_dense(torch.tensor(graph.A).to(dtype))
    print("ADJ matrix", adj_matrix)
    rowptr, col, _ = adj_matrix.csr()
    adj_matrix_dense = adj_matrix.to_dense()

    # PyG requires that the adj matrix is transposed when using SparseTensor.
    pred, att_weights = layer(x, adj_matrix.t(), return_attention_weights=True)
    print("pred", pred)
    loss = torch.sum(pred)
    pred.retain_grad()
    loss.backward()

    with torch.no_grad():
        result = compute_grads(x, adj_matrix_dense, layer.lin_src.weight,
                               layer.att_src, layer.att_dst,
                               layer.bias, att_weights.to_dense(), pred,
                               pred.grad)

    params = dict(layer.named_parameters())
    params['x'] = x

    for name, param in params.items():
        print(name, param.grad)
        if result[name] is not None:
            check_equal(param.grad, result[name].detach().numpy())
    print("x", x.grad)


if __name__ == '__main__':
    test_spmm_many_heads_gpu(3, 4, 2, 2137)
