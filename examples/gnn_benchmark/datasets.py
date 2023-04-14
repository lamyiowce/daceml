import functools
from typing import Tuple

import torch
import torch_geometric
from ogb import nodeproppred

dataset_classes = {
    'cora': functools.partial(torch_geometric.datasets.Planetoid, name='Cora'),
    'pubmed': functools.partial(torch_geometric.datasets.Planetoid,
                                name='PubMed'),
    'citeseer': functools.partial(torch_geometric.datasets.Planetoid,
                                  name='CiteSeer'),
    'flickr': torch_geometric.datasets.Flickr,
    'reddit': torch_geometric.datasets.Reddit,
}


def get_dataset_class(dataset_name: str):
    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]
    elif 'ogb' in dataset_name:
        return functools.partial(nodeproppred.PygNodePropPredDataset,
                                 name=dataset_name)
    raise NotImplementedError("No such dataset: ", dataset_name)


def get_dataset(dataset_name: str, device) -> torch_geometric.data.Data:
    data_path = f'/tmp/datasets/{dataset_name}'
    if dataset_name == 'small':
        _x = torch.tensor([[0., 1], [1, 1], [-1, 0]]).to(device)
        _edge_index = torch.tensor([[0, 0, 0, 2, 2], [0, 1, 2, 0, 2]]).to(device)
        _edge_attr = torch.tensor([1, 1, 1, 1., 1]).to(device)
        _y = torch.tensor([0, 1, 2]).to(device)
        data = torch_geometric.data.Data(x=_x, edge_index=_edge_index,
                                         edge_attr=_edge_attr, y=_y)
    else:
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(root=data_path)
        data = dataset[0]
        if device == torch.device('cuda'):
            data = data.pin_memory()
        data = data.to(device)
    return data
