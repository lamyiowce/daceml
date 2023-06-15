from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from examples.gnn_benchmark.report.plot_common import read_many_dfs, \
    DEFAULT_LABEL_MAP
from examples.gnn_benchmark.report.plot_performance import get_numbers


def make_roofline_plot(full_df, name, dataset: str, backward: bool,
                       labels=None, title=None):
    """Make a roofline plot for V100-PCIE-16GB."""
    assert 'single' in name
    df = get_numbers(full_df, dataset, backward=backward)
    y_df = df.pivot(index='Size', columns='Name', values='Performance')
    x_df = df.pivot(index='Size', columns='Name', values='Op intensity')

    ax = plt.figure(figsize=(8, 5))
    # Set both axes to logarithmic scaling.
    plt.xscale('log')
    plt.yscale('log')

    # Set axis limits.
    # plt.xlim(x_df.min().min() * 0.9, x_df.max().max()* 1.1)
    # plt.ylim(1e-1, 1e5)

    # Set axis labels.
    plt.xlabel('Operational Intensity [FLOP/Byte]')
    plt.ylabel('Performance [TFLOP/s]')

    # Plot the theoretical performance peak for V100.
    peak_v100 = 14  # https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
    mem_bound = 900 / 1024  # tb / s

    intersect_x = peak_v100 / mem_bound
    intersect_y = peak_v100

    plt.plot([0, intersect_x], [peak_v100, intersect_y], color='gray',
             linestyle='--',
             linewidth=1)
    plt.plot([intersect_x, 1e6], [intersect_y, peak_v100], color='black',
             linestyle='-',
             linewidth=1)

    plt.text(intersect_x * 1.2, intersect_y,
             f'Theoretical FP32: {peak_v100} TFLOP/s', rotation=0,
             verticalalignment='bottom',
             horizontalalignment='left', fontsize=12)

    # Plot the memory bound.
    plt.plot([0, intersect_x], [0, intersect_y], color='black', linestyle='-',
             linewidth=1)
    plt.plot([intersect_x, peak_v100 * 1.5],
             [intersect_y, mem_bound * peak_v100 * 1.5],
             color='gray', linestyle='--', linewidth=1)
    plt.text(1, mem_bound * 1.1, f'HBM bandwidth: {mem_bound * 1024} GB/s',
             rotation=45,
             rotation_mode='anchor',
             fontsize=12, transform_rotates_text=True)

    L2_mem_bound = 4198 / 1024
    plt.plot([0, peak_v100], [0, L2_mem_bound * peak_v100], color='gray',
             linestyle='--', linewidth=1)

    for col in x_df.columns:
        # Plot the different problem sizes with size labels at each point.
        plt.plot(x_df[col], y_df[col], label=DEFAULT_LABEL_MAP[col],
                 marker='o', markersize=5, linewidth=1, alpha=0.8)
        for i, row in x_df[col].iteritems():
            plt.annotate(f'{i}', (row, y_df[col][i]),
                         xytext=(row * 1.01, y_df[col][i] * 1.01), fontsize=8)

    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}-roofline.pdf')
    plt.show()


def main():
    cora_filenames = ['15.06.11.23-gcn_single_layer-cora-203685.csv']
    cora_df, cora_bwd_df = read_many_dfs(cora_filenames, backward=True)
    make_roofline_plot(cora_df, name='gcn-single-cora', dataset='cora',
                       backward=False, title='GCN Cora FWD')

    arxiv_filenames = ['15.06.12.59-gcn_single_layer-ogbn-arxiv-203691.csv']
    arxiv_df, arxiv_bwd_df = read_many_dfs(arxiv_filenames, backward=True)
    make_roofline_plot(arxiv_df, name='gcn-single-arxiv', dataset='arxiv',
                       backward=False, title='GCN OGB Arxiv FWD')


if __name__ == '__main__':
    main()
