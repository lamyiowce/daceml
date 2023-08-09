import seaborn as sns
from matplotlib import pyplot as plt, colors

from examples.gnn_benchmark.report.plot_common import read_many_dfs, PLOT_FOLDER, get_colors


def main():
    plot_gcn_memory()
    # plot_gat_memory()
    plot_gcn_caching_comparison()


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
    make_memory_plot(df, label_map, name, columns=['Backward'])


def plot_gat_memory():
    # datalist = ['04.08.08.57-gat-ogbn-arxiv-233383-memory.csv']
    # datalist = ['01.08.13.57-gat-ogbn-arxiv-228542-memory.csv']
    datalist = ['04.08.10.18-gat-ogbn-arxiv-233429-memory.csv']
    df, _ = read_many_dfs(datalist, backward=False)
    df.drop_duplicates(subset=['Name', 'Size'], inplace=True)

    label_map = {
        'coo_cached': 'Cache features, edge mask, edge weights',
        'coo_cached_feat_and_alpha': 'Cache features and node attention',
        'coo_cached:coo_cached_feat_only': 'Cache only features',
        'coo': 'No caching',
    }

    for col in ['Forward', 'Forward with grads', 'Backward']:
        df[col] = df[col] / 1024 / 1024  # TO MB

    name = 'GAT Memory Usage'
    order = ['coo_cached', 'coo_cached_feat_and_alpha', 'coo_cached:coo_cached_feat_only', 'coo']
    # df.sort_values(by="Name", key=lambda column: column.map(lambda e: order.index(e)), inplace=True)
    make_memory_plot(df, label_map, name, columns=['Forward with grads', 'Backward'], order=order,
                     sizes=[8, 16, 32, 64, 128])


def pretty_print_megabytes(num_megabytes: int):
    if num_megabytes < 1024:
        return f"{num_megabytes:.2f} MB"
    else:
        return f"{num_megabytes / 1024:.2f} GB"


def plot_gcn_caching_comparison():
    datalist = ['01.08.14.31-gcn_single_layer-ogbn-arxiv-228575-memory.csv']
    mem_df, _ = read_many_dfs(datalist, backward=False)
    mem_df['Name'] = mem_df['Name'].map(lambda x: 'dace_' + x)
    datalist = ['01.08.14.31-gcn_single_layer-ogbn-arxiv-228575.csv']
    fwd_df, bwd_df = read_many_dfs(datalist, backward=True)

    df = mem_df.merge(bwd_df, on=['Name', 'Size', 'Model'], suffixes=(' Bwd', ' Mem'))
    print(df)

    mem_df = df.pivot(index='Size', columns='Name', values='Backward')
    mem_df['dace_csc_adapt_cached'] = mem_df['dace_csc_adapt_cached'] / mem_df['dace_csc_adapt']

    speedup_df = df.pivot(index='Size', columns='Name', values='Median')
    speedup_df['dace_csc_adapt_cached'] = speedup_df['dace_csc_adapt'] / speedup_df['dace_csc_adapt_cached']

    ax = plt.figure(figsize=(6, 3)).gca()
    # Plot speedup and memory use on the same plot.

    # Make the line a bit transparent but not the marker.
    ax.plot(range(len(speedup_df.index)), mem_df['dace_csc_adapt_cached'], label='Relative memory use',
            linestyle='--', marker='x', markeredgecolor=colors.to_rgba('tab:orange', 1.0),
            color=colors.to_rgba('tab:orange', 0.5))
    # use left y axis for memory use.
    ax.set_ylabel('Relative memory use')
    ax.set_ylim(0.0, 1.4)

    # Make twin axes aligned.
    ax2 = ax.twinx()
    ax2.set_ylabel('Speedup')
    ax2.set_ylim(0.0, 1.4)

    ax2.plot(range(len(speedup_df.index)), speedup_df['dace_csc_adapt_cached'], label='Speedup', marker='o',
            linestyle=':', markeredgecolor=colors.to_rgba('tab:blue', 1.0),
            color=colors.to_rgba('tab:blue', 0.5))
    # ax.set_xscale('log')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.xticks(range(len(speedup_df.index)), [str(x) for x in speedup_df.index])
    # ax.set_xticks([str(x) for x in mem_df.index])
    # Remove top border.
    ax2.spines['top'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Add one legend for both.
    ax.set_xlabel('Output feature size')
    ax2.set_xlabel('Output feature size')
    lines_labels = [ax.get_legend_handles_labels() for ax in (ax, ax2)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    plt.legend(lines, labels, loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / 'thesis' / 'gcn_caching_comparison.pdf', bbox_inches='tight')
    plt.show()


def make_memory_plot(df, label_map, name, columns, order=None, sizes=None):
    if sizes is not None:
        df = df[df['Size'].isin(sizes)]
    plt.rcParams.update({'font.size': 12})
    figsize = (8, 4 * len(columns))
    plt.rcParams['figure.figsize'] = figsize
    fig, axs = plt.subplots(nrows=len(columns), squeeze=False)
    for ax, column in zip(axs[0], columns):
        sns.barplot(data=df, x='Size', y=column, hue='Name', hue_order=order,
                    edgecolor=(1.0, 1.0, 1.0, 0.4),
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
        # Set yticks to powers of two
        ax.set_yticks([i for i in range(0, int(df[column].max()), 1024)])
        ax.set_yticklabels([pretty_print_megabytes(int(y)) for y in ax.get_yticks()])
        ax.set_xlabel('')
        if len(columns) > 1:
            ax.set_title(column.split(' ')[0])
        # remove the top and right spines from the plot
        sns.despine()
    plt.xlabel("Output feature size")
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
