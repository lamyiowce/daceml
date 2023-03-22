import statistics

import matplotlib.pyplot as plt
import numpy as np
import tabulate

from examples.gnn_benchmark.datasets import get_dataset


def get_info(name, data, degree):
    return [
        name,
        data.x.shape[0],
        data.edge_index.shape[1],
        data.edge_index.shape[1] / (data.x.shape[0] * data.x.shape[0]),
        data.x.shape[1],
        data.y.max().item() + 1,
        degree.mean(),
        statistics.median(degree),
        statistics.stdev(degree)
    ]


def print_info(rows):
    headers = ['Name', 'Num Nodes', 'Num Edges', '% NNZ', 'Num node features', 'Num classes', 'avg degree', 'meidan degree', 'std degree']
    print(tabulate.tabulate(rows,
                            headers=headers,
                            floatfmt='.4f',
                            tablefmt='github'))


def visualize_single_dataset(ax, data):
    dataset, num_node_features, num_classes = get_dataset(data, 'cpu')
    print(dataset)
    degree = compute_degrees(dataset)
    print(degree.max())
    ax.hist(degree, bins=100)
    ax.set_title(f'{data}: Node degree distribution')
    # Set log scale
    ax.set_yscale('log')
    # Add grid
    ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    return get_info(data, dataset, degree)


def visualize_datasets():
    datasets = ['flickr', 'cora', 'pubmed', 'citeseer']
    # Crete a figure with margins
    fig, axes = plt.subplots(len(datasets) // 2 + len(datasets) % 2, 2, figsize=(10, 10))
    rows = []
    for ax, data in zip(axes.flatten(), datasets):
        rows += [visualize_single_dataset(ax, data)]

    print_info(rows)
    fig.tight_layout()
    plt.savefig(f'degrees.pdf', bbox_inches='tight')
    plt.show()


def compute_degrees(dataset):
    edge_index = dataset.edge_index
    degree = np.zeros((dataset.x.shape[0]))
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1
        degree[edge_index[1, i]] += 1
    return degree


if __name__ == '__main__':
    visualize_datasets()
