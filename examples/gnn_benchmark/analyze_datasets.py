import statistics

import click
import matplotlib.pyplot as plt
import numpy as np
import tabulate

from examples.gnn_benchmark import sparse
from examples.gnn_benchmark.datasets import get_dataset


def get_info(name, data, degree):
    return [
        name,
        data.x.shape[0],
        data.edge_index.shape[1],
        100 * data.edge_index.shape[1] / (data.x.shape[0] * data.x.shape[0]),
        data.x.shape[1],
        data.y.max().item() + 1,
        degree.mean(),
        statistics.median(degree),
        statistics.stdev(degree)
    ]


def print_info(rows):
    headers = ['Name', 'Num Nodes', 'Num Edges', '% NNZ', 'Num node features',
               'Num classes', 'avg degree', 'Median degree', 'Std degree']
    print(tabulate.tabulate(rows,
                            headers=headers,
                            floatfmt='.6f',
                            tablefmt='github'))


def visualize_single_dataset(ax, data):
    dataset = get_dataset(data, 'cpu')
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


@click.command()
@click.argument('datasets', nargs=-1)
def visualize_datasets(datasets):
    # Crete a figure with margins
    fig, axes = plt.subplots(len(datasets) // 2 + len(datasets) % 2, 2,
                             figsize=(10, 10))
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


def check_coalescence(data, name):
    print(name)
    # Convert to CSR.
    # csr_data = sparse.CsrGraph.from_pyg_data(data, compute_input_grad=False)
    #
    # total_hits = 0
    # cache_line_size = 32
    # col_div = csr_data.columns // cache_line_size
    # for i in range(len(csr_data.rowptrs) - 1):
    #     values, counts = np.unique(col_div[csr_data.rowptrs[i]:csr_data.rowptrs[i + 1]], return_counts=True)
    #     total_hits += np.sum(counts - 1)
    #
    # print(f'Cache hits horizontal: {total_hits}, {total_hits / csr_data.columns.shape[0] * 100:.2f}%')
    #
    # csc_data = sparse.CscGraph.from_pyg_data(data, compute_input_grad=False)
    #
    # total_hits = 0
    # cache_line_size = 32
    # col_div = csc_data.rows // cache_line_size
    # for i in range(len(csc_data.colptrs) - 1):
    #     values, counts = np.unique(col_div[csc_data.colptrs[i]:csc_data.colptrs[i + 1]], return_counts=True)
    #     total_hits += np.sum(counts - 1)
    #
    # print(f'Cache hits vertical: {total_hits}, {total_hits / csc_data.rows.shape[0] * 100:.2f}%')

    coo_data = sparse.CooGraph.from_pyg_data(data, compute_input_grad=False)

    # Check if sorted.
    last_row = -1
    last_col = -1
    sorted = True
    for i in range(coo_data.rows.shape[0]):
        col = coo_data.cols[i]
        row = coo_data.rows[i]
        if row < last_row:
            sorted = False
            break
        else:
            if row == last_row and col < last_col:
                sorted = False
                break
        last_row = row
        last_col = col
    print("Sorted" if sorted else "Not sorted")

    # Check if sorted.
    last_row = -1
    last_col = -1
    sorted = True
    for i in range(coo_data.rows.shape[0]):
        col = coo_data.cols[i]
        row = coo_data.rows[i]
        if col < last_col:
            sorted = False
            break
        if col == last_col and row < last_row:
            sorted = False
            break
        last_row = row
        last_col = col
    print("Sorted" if sorted else "Not sorted")


if __name__ == '__main__':
    # visualize_datasets()
    check_coalescence(get_dataset('cora', 'cpu'), 'cora')
    check_coalescence(get_dataset('ogbn-arxiv', 'cpu'), 'ogbn-arxiv')
    check_coalescence(get_dataset('citeseer', 'cpu'), 'citeseer')
    check_coalescence(get_dataset('pubmed', 'cpu'), 'pubmed')
    check_coalescence(get_dataset('flickr', 'cpu'), 'flickr')
    check_coalescence(get_dataset('reddit', 'cpu'), 'reddit')
