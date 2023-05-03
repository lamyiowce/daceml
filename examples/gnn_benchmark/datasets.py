import functools

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


def get_dataset(dataset_name: str, device, val_dtype: torch.dtype = torch.float32) -> torch_geometric.data.Data:
    data_path = f'/tmp/datasets/{dataset_name}'
    if dataset_name == 'small':
        _x = torch.tensor([[0., 1], [1, 1], [-1, 0], [3, 5]]).to(device).to(val_dtype)
        _edge_index = torch.tensor([[0, 0, 0, 3, 1], [0, 1, 2, 0, 1]]).to(device)
        _edge_attr = torch.tensor([1, 1, 1, 1., 1]).to(device).to(val_dtype)
        _y = torch.tensor([0, 1, 2, 1]).to(device)
        data = torch_geometric.data.Data(x=_x, edge_index=_edge_index,
                                         edge_attr=_edge_attr, y=_y)
    else:
        dataset_class = get_dataset_class(dataset_name)
        dataset = dataset_class(root=data_path)
        data = dataset[0]
        data.y = data.y.squeeze().contiguous()
        if device == torch.device('cuda'):
            data = data.pin_memory()
        data = data.to(device)
        data.x = data.x.to(val_dtype)
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.to(val_dtype)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(val_dtype)
        print(data)
    return data
