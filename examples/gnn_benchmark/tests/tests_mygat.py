import copy

import pytest
import torch

from examples.gnn_benchmark.tests.common import check_equal, check_grads, pred_torch, setup_data


class MyGat(torch.nn.Module):
    def __init__(self, weight, att_src, att_dst, bias=None):
        super().__init__()
        self.weight = copy.deepcopy(weight)
        self.att_src = copy.deepcopy(att_src)
        self.att_dst = copy.deepcopy(att_dst)
        if bias is not None:
            self.bias = copy.deepcopy(bias)
        self.heads = att_src.shape[1]

    def forward(self, x, adj_matrix):
        N = x.shape[0]
        heads = self.heads
        F_out = self.weight.shape[0] // heads
        self.H_prime = x @ self.weight.t()  # N x H * F_out
        self.H_prime = torch.reshape(self.H_prime, (N, self.heads, F_out))
        alpha_src = torch.sum(self.H_prime * self.att_src, dim=-1)  # N x H
        alpha_dst = torch.sum(self.H_prime * self.att_dst, dim=-1)  # N x H
        C = (alpha_src[None, :] + alpha_dst[:, None])  # N x N x H
        Tau = adj_matrix.t()[..., None] * torch.exp(torch.maximum(0.2 * C, C))
        Tau_sum = torch.sum(Tau, dim=1)[:, None]  # N x 1 x H
        self.att_weights = Tau / Tau_sum  # N x N x H
        # Z = (adj_matrix.t()[..., None] * self.att_weights) @ self.H_prime  # N x H x F_out
        Z = torch.einsum('mnh,nhf->mhf', (adj_matrix.t()[..., None] * self.att_weights), self.H_prime)
        return torch.reshape(Z, (N, heads * F_out)) + self.bias


@pytest.mark.parametrize("heads", [1, 2])
def test_mygat(heads):
    N = 3
    F_in = 2
    F_out = 4

    adj_matrix, layer, random_mask, x = setup_data(N, F_in, F_out, heads)
    adj_matrix_dense = adj_matrix.to_dense()

    att_weights, pred = pred_torch(adj_matrix, layer, random_mask, x)

    mygat = MyGat(weight=layer.lin_src.weight, att_src=layer.att_src,
                  att_dst=layer.att_dst, bias=layer.bias)
    mygat_x = copy.deepcopy(x)
    mygat_x.grad = None
    mygat_out = mygat.forward(mygat_x, adj_matrix_dense)
    assert torch.allclose(mygat_out, pred)
    assert check_equal(mygat.att_weights.detach().numpy(),
                       att_weights.to_dense().detach().numpy())
    loss = torch.sum(mygat_out * random_mask)
    mygat.H_prime.retain_grad()
    mygat.att_weights.retain_grad()
    loss.backward()

    params = dict(layer.named_parameters())
    params['x'] = x

    mygat_params = dict(mygat.named_parameters())
    mygat_params['x'] = mygat_x
    mygat_params['lin_src.weight'] = mygat_params['weight']

    check_grads(expected_params=params, result=mygat_params)
