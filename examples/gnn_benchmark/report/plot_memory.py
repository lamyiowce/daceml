import seaborn as sns
from matplotlib import pyplot as plt

from examples.gnn_benchmark.report.plot_common import read_many_dfs, PLOT_FOLDER, get_colors


def main():
    # plot_gcn_memory()
    plot_gat_memory()


def prep_df_memory(full_df, column, col_order=None):
    dupl_cols = ['Size', 'Model', 'Name']
    full_df = full_df.drop_duplicates(subset=dupl_cols)
    df_fwd = full_df.pivot(index=column, columns='Name', values='Forward')
    df_fwd_grads = full_df.pivot(index=column, columns='Name', values='Forward with grads')
    df_bwd = full_df.pivot(index=column, columns='Name', values='Backward')
    dfs = [df_fwd, df_fwd_grads, df_bwd]
    if col_order is None:
        col_order = sorted(df_fwd.columns,
                           key=lambda x: ('torch' in x, 'edge_list' in x))

    dfs = [df.reindex(col_order, axis=1) for df in dfs]
    return dfs


def plot_gcn_memory():
    datalist = ['01.08.14.31-gcn_single_layer-ogbn-arxiv-228575-memory.csv']
    df, _ = read_many_dfs(datalist, backward=False)

    label_map = {
        'csc': 'FWD: Transform-first, BWD: fused-propagate',
        'csc_alt': 'FWD: Propagate-first, BWD: split-propagate',
        'csc_adapt': 'FWD & BWD: Adaptive, no caching',
        'csc_cached': 'FWD: Propagate-first, BWD: split-propagate, with caching',
        'csc_adapt_cached': 'FWD & BWD: Adaptive, with caching',
    }

    for col in ['Forward', 'Forward with grads', 'Backward']:
        df[col] = df[col] / 1024 / 1024  # TO MB

    name = 'GCN Memory Usage'
    make_memory_plot(df, label_map, name, columns=['Forward', 'Backward'])


def plot_gat_memory():
    datalist = ['01.08.13.57-gat-ogbn-arxiv-228542-memory.csv']
    df, _ = read_many_dfs(datalist, backward=False)

    label_map = {
        'dace_coo_cached': 'Full caching',
        'dace_coo_cached_feat_and_alpha': 'Cache features and node attention',
        'dace_coo_cached:coo_cached_feat_only': 'Cache only features',
        'dace_coo': 'No caching',
    }

    for col in ['Forward', 'Forward with grads', 'Backward']:
        df[col] = df[col] / 1024 / 1024  # TO MB

    name = 'GAT Memory Usage'
    make_memory_plot(df, label_map, name, columns=['Forward', 'Backward'])


def pretty_print_megabytes(num_megabytes: int):
    if num_megabytes < 1024:
        return f"{num_megabytes:.2f} MB"
    else:
        return f"{num_megabytes / 1024:.2f} GB"


def make_memory_plot(df, label_map, name, columns):
    plt.rcParams.update({'font.size': 12})
    figsize = (8, 8)
    plt.rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(nrows=2)
    for ax, column in zip(axs, columns):
        sns.barplot(data=df, x='Size', y=column, hue='Name', edgecolor=(1.0, 1.0, 1.0, 0.4),
                    palette=get_colors(df['Name']), ax=ax)
        ax.set_axisbelow(True)
        ax.set_ylim(ymax=max(df[column].max() * 1.2, 0.9))
        ax.yaxis.grid(color='lightgray', linestyle='--')
        ax.set_ylabel("Memory [MB]")
        plt.xticks(rotation=0)
        for container in ax.containers:
            if hasattr(container, 'patches'):
                ax.bar_label(container, fmt=pretty_print_megabytes, fontsize=7, zorder=10,
                             rotation=90,
                             label_type='edge', padding=3,
                             )

        bars = ax.patches
        patterns = ('\\\\\\\\\\', '/////', '|||||', '....', 'xxxx', '++++')[
                   :len(df['Name'].unique())]
        hatches = [p for p in patterns for _ in range(len(df['Size'].unique()))]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        # plt.legend(labels, loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        labels = [label_map.get(name, name) for name in labels]
        ax.legend(handles, labels)
        ax.set_xlabel('')
        ax.set_title(column)
    plt.xlabel("Hidden size")
    plt.tight_layout()
    # put today's date in the filename
    clean_name = name.replace(',', '').replace('+', '').replace('  ', ' ').replace(':', '')
    path = PLOT_FOLDER / 'thesis' / f'{clean_name}.pdf'
    plt.savefig(path, bbox_inches='tight')
    # plt.title(name.upper())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
