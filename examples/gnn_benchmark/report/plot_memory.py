import seaborn as sns
from matplotlib import pyplot as plt

from examples.gnn_benchmark.report.plot_common import read_many_dfs, make_plot, DEFAULT_LABEL_MAP, \
    PLOT_FOLDER, get_colors


def main():
    plot_gcn_memory()


# def make_plot(full_df, name, plot_column, label_map=None, bwd_df=None, legend_outside=False,
#               skip_timestamp=False, xlabel=None, color_map=None, col_order=None):
#     plt.rcParams.update({'font.size': 13})
#     df, std_df = prep_df(full_df, column=plot_column, col_order=col_order)
#     colors = color_map or get_colors(df.columns)
#     bar_width = 0.85
#     figsize = (1.5 + len(df) * 0.9, 6.)
#     ax = df.plot(figsize=figsize, kind='bar', ylabel='Runtime [ms]',
#                  xlabel=xlabel or COLUMN_PRETTY_NAMES[plot_column], color=colors,
#                  yerr=std_df, label='Forward', width=bar_width)
#
#     ax.set_axisbelow(True)
#     ax.yaxis.grid(color='lightgray', linestyle='--')
#     if bwd_df is not None and bwd_std_df is not None:
#         ax.set_ylim(ymax=max((bwd_df + bwd_std_df).max().max() * 1.09, 0.9))
#     else:
#         ax.set_ylim(ymax=max((df + std_df).max().max() * 1.09, 0.9))
#     ax.set_ylabel("Runtime [ms]")
#     ax.set_xlabel(xlabel or COLUMN_PRETTY_NAMES[plot_column])
#     # plt.title(name.upper())
#
#     default_label_map = {
#         'torch_csr': 'Torch CSR',
#         'torch_edge_list': 'Torch Edge List',
#         'compiled_torch_edge_list': 'Torch Edge List (compiled)',
#     }
#     default_label_map.update(label_map or {})
#     labels = [default_label_map.get(name, name) for name in df.columns]
#
#
#     # ax.legend(legend_handles[::-1], labels[::-1])
#
#     plt.xticks(rotation=0)
#
#     for container in ax.containers:
#         if hasattr(container, 'patches'):
#             if container.patches[0].get_facecolor()[-1] < 1.0:
#                 padding = 6
#                 label_type = 'center'
#             else:
#                 padding = 6
#                 label_type = 'edge'
#             # Set text size.
#             # Make the labels appear on top z.
#             ax.bar_label(container, fmt="%.2f", padding=padding, fontsize=7, zorder=10, rotation=90, label_type=label_type)
#
#     bars = ax.patches
#     patterns = ('\\\\\\\\\\', '/////', '|||||', '....', 'xxxx', '++++')
#     hatches = [p for p in patterns for i in range(len(df))]
#     for bar, hatch in zip(bars, hatches):
#         if bar.get_facecolor()[-1] == 1.0:
#             bar.set_hatch(hatch)
#
#     if legend_outside:
#         plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
#     else:
#         plt.legend(labels, loc='upper left')
#
#     plt.tight_layout()
#     # put today's date in the filename
#     clean_name = name.replace(',', '').replace('+', '').replace('  ', ' ').replace(':', '')
#
#     if skip_timestamp:
#         path = PLOT_FOLDER / 'thesis' / f'{clean_name}.pdf'
#     else:
#         path = PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} {clean_name}.pdf'
#
#     plt.savefig(path, bbox_inches='tight')
#
#     plt.title(name.upper())
#     plt.tight_layout()
#     plt.show()


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

    bwd_label_map = {
        'csc': 'FWD: Transform-first, BWD: fused-propagate',
        'csc_alt': 'FWD: Propagate-first, BWD: split-propagate',
        'csc_adapt': 'Adaptive, no caching',
        'csc_cached': 'FWD: Propagate-first, BWD: split-propagate, with caching',
        'csc_adapt_cached': 'Adaptive with caching',
    }
    forward_label_map = {
        'csc': 'FWD: Transform-first',
        'csc_alt': 'FWD: Propagate-first',
        'csc_adapt': 'Adaptive, no caching',
        'csc_cached': 'FWD: Propagate-first with caching',
        'csc_adapt_cached': 'Adaptive with caching',
    }
    for col, label_map in [('Forward', forward_label_map),
                           ('Forward with grads', bwd_label_map),
                           ('Backward', bwd_label_map)]:
        df[col] = df[col] / 1024 / 1024 # TO MB
        name = 'GCN Memory Usage: ' + col
        make_memory_plot(df, label_map, name, column=col)


def make_memory_plot(df, label_map, name, column):
    ax = sns.barplot(data=df, x='Size', y=column, hue='Name', edgecolor=(1.0, 1.0, 1.0, 0.4),
                     palette=get_colors(df['Name']))
    ax.set_axisbelow(True)
    ax.set_ylim(ymax=max(df[column].max() * 1.09, 0.9))
    ax.yaxis.grid(color='lightgray', linestyle='--')
    ax.set_ylabel("Memory [MB]")
    ax.set_xlabel("Hidden size")
    plt.xticks(rotation=0)
    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f MB", fontsize=7, zorder=10, rotation=90,
                         label_type='center'
                         )

    bars = ax.patches
    patterns = ('\\\\\\\\\\', '/////', '|||||', '....', 'xxxx', '++++')
    hatches = patterns * 8
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    # plt.legend(labels, loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    labels = [label_map.get(name, name) for name in labels]
    ax.legend(handles, labels)
    plt.tight_layout()
    # put today's date in the filename
    clean_name = name.replace(',', '').replace('+', '').replace('  ', ' ').replace(':', '')
    path = PLOT_FOLDER / 'thesis' / f'{clean_name}.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.title(name.upper())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
