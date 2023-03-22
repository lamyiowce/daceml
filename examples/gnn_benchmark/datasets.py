from typing import Tuple

import torch
import torch_geometric


def get_dataset(dataset_name: str, device) -> Tuple[
    torch_geometric.data.Data, int, int]:
    if dataset_name == 'cora':
        dataset = torch_geometric.datasets.Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0].to(device)
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
    elif dataset_name == 'small':
        _x = torch.tensor([[0., 1], [1, 1], [-1, 0]]).to(device)
        _edge_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0,
                                                      2]]).to(device)
        _edge_attr = torch.tensor([1, 1, 1, 1., 1]).to(device)
        data = torch_geometric.data.Data(x=_x, edge_index=_edge_index, edge_attr=_edge_attr)
        num_node_features = _x.shape[1]
        num_classes = 2
    else:
        raise NotImplementedError("No such dataset: ", dataset_name)
    return data, num_node_features, num_classes
