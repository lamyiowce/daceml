import pandas as pd
from matplotlib import pyplot as plt

from examples.gnn_benchmark.report.plot import get_colors
from examples.gnn_benchmark.report.plot_common import read_many_dfs, \
    PLOT_FOLDER, DEFAULT_LABEL_MAP

fwd_filenames = ['gcn-numbers.csv']
bwd_filenames = ['gcn-numbers-bwd.csv']


def get_numbers(df: pd.DataFrame, dataset: str, backward: bool = False):
    assert dataset in ['cora', 'arxiv']
    filenames = bwd_filenames if backward else fwd_filenames
    df_list = []
    for filename in filenames:
        df_list.append(pd.read_csv(filename))
    numbers_df = pd.concat(df_list)
    numbers_df = numbers_df[numbers_df['Dataset'] == dataset]

    df = df[df['Name'].str.contains('dace')]
    df['Impl'] = df['Name'].str.replace('dace_', '')
    # Match entries from the numbers_df to the df based on Name, model, hidden
    # size.
    df = pd.merge(df, numbers_df, on=['Impl', 'Model', 'Size'], how='inner',
                  suffixes=('', '_numbers'))
    df['Performance'] = df['Flops'] / df['Median'] * 1000  # (Time is in ms.)
    df['Performance'] = df['Performance'] / 1e12  # TFLOP/s
    return df


def make_performance_plot(full_df, name, dataset: str, backward: bool,
                          labels=None, title=None):
    assert 'single' in name
    df = get_numbers(full_df, dataset, backward=backward)
    df = df.pivot(index='Size', columns='Name', values='Performance')


    colors = get_colors(df.columns)
    default_labels = DEFAULT_LABEL_MAP.copy()
    default_labels.update(labels or {})
    labels = default_labels

    ax = df.plot(figsize=(5, 5), kind='barh', ylabel='Runtime [ms]',
                       xlabel='Hidden size', color=colors)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='lightgray', linestyle='--')
    ax.set_xlabel("Performance [TFLOP / s]")
    ax.set_ylabel("Hidden size")

    labels = [labels.get(name, name) for name in df.columns]
    plt.legend(labels, loc='lower center')
    peak_v100 = 14  # https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
    ax.plot([peak_v100, peak_v100], [-10, 10], color='black', linestyle='--',
            linewidth=1)
    ax.text(peak_v100 * 0.95, 1, f'V100 peak: {peak_v100} TFLOP / s',
            rotation=90,
            verticalalignment='center',
            horizontalalignment='left', fontsize=12)
    ax.set_xlim(xmax=peak_v100 * 1.1)
    plt.title(title or name.upper())
    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(
        PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} {name}-performance.pdf',
        bbox_inches='tight')
    plt.show()


def plot_performance_gcn():
    cora_filenames = ['15.06.11.23-gcn_single_layer-cora-203685.csv']
    cora_df, cora_bwd_df = read_many_dfs(cora_filenames, backward=True)
    make_performance_plot(cora_df, 'gcn-single-cora', dataset='cora',
                          backward=False, title='Performance: GCN FWD, Cora')
    make_performance_plot(cora_bwd_df, 'gcn-single-cora', dataset='cora',
                          backward=True, title='Performance: GCN BWD, Cora')

    arxiv_filenames = ['15.06.12.59-gcn_single_layer-ogbn-arxiv-203691.csv']
    arxiv_df, arxiv_bwd_df = read_many_dfs(arxiv_filenames, backward=True)
    make_performance_plot(arxiv_df, 'gcn-single-arxiv', dataset='arxiv',
                          backward=False, title='Performance: GCN FWD, OGB Arxiv')
    make_performance_plot(arxiv_bwd_df, 'gcn-single-arxiv', dataset='arxiv',
                          backward=True, title='Performance: GCN BWD, OGB Arxiv')


def main():
    plot_performance_gcn()


if __name__ == '__main__':
    main()
