from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

DATA_FOLDER = Path(__file__).parent / 'data'

DEFAULT_LABEL_MAP = {
    'torch_csr': 'Torch CSR',
    'torch_edge_list': 'Torch Edge List',
    'torch_dgnn': 'Torch DGNN-GAT',
    'dace_csr': 'DaCe CSR',
    'dace_csr_adapt': 'DaCe CSR, adapt MM order',
    'dace_coo': 'DaCe COO',
    'dace_coo_adapt': 'DaCe COO, adapt MM order',
    'dace_csc': 'DaCe CSC',
    'dace_csc_adapt': 'DaCe CSC (adapt MM order)',
    'dace_csr_coo_adapt': 'DaCe CSR/COO adapt, CSR 50%',
    'dace_csr_coo_adapt-0.99': 'DaCe CSR/COO adapt, CSR 99%',
    'dace_csr_coo': 'DaCe CSR/COO, CSR 50%',
    'dace_coo_cached': 'DaCe COO, cached',
    'dace_coo_cached:coo_cached_feat_only': 'DaCe COO, cached features',
    'dace_coo_cached_feat_and_alpha': 'DaCe COO, cached features, node attention',
    'dace_coo_adapt_cached': 'DaCe COO (adapt, cached)',
    'dace_coo_stable_cached:coo_cached': 'DaCe COO (cached)',
    'dace_coo_stable:coo': 'DaCe COO',
    'dace_csc_cached': 'DaCe CSC (cached)',
    'dace_csc_adapt_cached': 'DaCe CSC (adapt, cached)',
}

DEFAULT_LABEL_MAP.update(
    {f'dace_csr_coo_adapt-0.{i}': f'Dace CSR/COO adapt, CSR {i}0%' for i in
     range(3, 9)})

DEFAULT_LABEL_MAP.update(
    {f'{key}_compiled': f'{name} (compiled)' for key, name in
     DEFAULT_LABEL_MAP.items() if
     'torch' in key})

DEFAULT_LABEL_MAP.update(
    {name[5:]: val for name, val in DEFAULT_LABEL_MAP.items() if
     name.startswith('dace_')})

COLUMN_PRETTY_NAMES = {
    'Size': 'Hidden size',
    'Num Features': 'Input features size',
    'Num Layers': 'Number of layers',
}

PLOT_FOLDER = Path(__file__).parent / 'plots'
MODELING_FOLDER = Path(__file__).parent / 'modeling'


def read_many_dfs(filenames, name_to_replace=None, backward: bool = True,
                  names=None, name_fns=None):
    dfs = []
    bwd_dfs = []
    names = names or filenames
    name_fns = name_fns or [None] * len(filenames)
    assert len(names) == len(filenames)
    assert len(name_fns) == len(filenames)
    for filename, name, name_fn in zip(filenames, names, name_fns):
        df_temp = pd.read_csv(DATA_FOLDER / filename, comment='#')
        if name_to_replace:
            dace_rows = df_temp['Name'].str.contains(name_to_replace)
            df_temp.loc[dace_rows, 'Name'] = df_temp.loc[
                dace_rows, 'Name'].apply(
                name_fn) if name_fn else name
        dfs.append(df_temp)
        if backward:
            bwd_path = DATA_FOLDER / filename.replace('.csv', '-bwd.csv')
            df_temp = pd.read_csv(bwd_path, comment='#')
            if name_to_replace:
                dace_rows = df_temp['Name'].str.contains(name_to_replace)
                df_temp.loc[dace_rows, 'Name'] = df_temp.loc[
                    dace_rows, 'Name'].apply(
                    name_fn) if name_fn else name
            bwd_dfs.append(df_temp)

    return pd.concat(dfs), pd.concat(bwd_dfs) if backward else None


def get_colors(names: pd.Series):
    reds_intense = ['gold', 'orange', 'darkorange']
    reds = ['indianred', 'lightcoral', 'lightsalmon', 'pink']
    greens = ['olivedrab', 'yellowgreen', 'forestgreen', 'limegreen',
              'seagreen', 'mediumseagreen',
              'lightseagreen', 'mediumturquoise', 'paleturquoise', 'steelblue',
              'lightskyblue'] * 2
    # Get unique names. If 'torch' is in the name, use red, otherwise green.
    unique_names = names.unique()
    color_dict = {}
    for name in unique_names:
        if 'compile' in name:
            color_dict[name] = reds_intense.pop(0)
        elif 'torch' in name:
            color_dict[name] = reds.pop(0)
        else:
            color_dict[name] = greens.pop(0)
    return names.map(color_dict)


def prep_df(full_df, column, col_order=None):
    if 'Num Layers' in full_df.columns:
        dupl_cols = ['Size', 'Model', 'Name', 'Num Features', 'Num Layers']
    else:
        dupl_cols = ['Size', 'Model', 'Name']
    full_df = full_df.drop_duplicates(subset=dupl_cols)
    df = full_df.pivot(index=column, columns='Name', values='Median')
    std_df = full_df.pivot(index=column, columns='Name', values='Stdev')
    if col_order is not None:
        df = df.reindex(col_order, axis=1)
        std_df = std_df.reindex(col_order, axis=1)
    else:
        sorted_cols = sorted(df.columns,
                             key=lambda x: ('torch' in x, 'edge_list' in x))
        df = df.reindex(sorted_cols, axis=1)
        std_df = std_df.reindex(sorted_cols, axis=1)
    return df, std_df


def make_plot(full_df, name, plot_column, label_map=None, bwd_df=None, legend_outside=False,
              skip_timestamp=False, xlabel=None, color_map=None, col_order=None, figsize=None):
    plt.rcParams.update({'font.size': 13})
    df, std_df = prep_df(full_df, column=plot_column, col_order=col_order)
    colors = color_map or get_colors(df.columns)
    bar_width = 0.85
    figsize = figsize or (1.5 + len(df) * 0.9, 6)
    if bwd_df is None:
        ax = df.plot(figsize=figsize, kind='bar', ylabel='Runtime [ms]',
                     xlabel=xlabel or COLUMN_PRETTY_NAMES[plot_column], color=colors,
                     yerr=std_df, label='Forward', width=bar_width)
    else:
        bwd_df, bwd_std_df = prep_df(bwd_df, plot_column, col_order=col_order)
        if len(bwd_df.columns) != len(df.columns):
            print('Warning: bwd_df and df have different lengths')
            print('Differing columns: ', set(bwd_df.columns) ^ set(df.columns))
        ax = bwd_df.plot(figsize=figsize,
                         kind='bar',
                         color=colors,
                         yerr=bwd_std_df,
                         label='Backward',
                         width=bar_width,
                         edgecolor=(1.0, 1.0, 1.0, 0.4),
                         error_kw=dict(ecolor='gray', lw=1, capsize=1, capthick=1))
        df.plot(kind='bar',
                ylabel='Runtime [ms]',
                xlabel=xlabel or COLUMN_PRETTY_NAMES[plot_column],
                color='black',
                alpha=0.2,
                yerr=std_df,
                ax=ax,
                label='Forward',
                error_kw=dict(ecolor='gray', lw=1, capsize=1, capthick=1),
                width=bar_width)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='lightgray', linestyle='--')
    if bwd_df is not None and bwd_std_df is not None:
        ax.set_ylim(ymax=max((bwd_df + bwd_std_df).max().max() * 1.1, 0.9))
    else:
        ax.set_ylim(ymax=max((df + std_df).max().max() * 1.1, 0.9))
    ax.set_ylabel("Runtime [ms]")
    ax.set_xlabel(xlabel or COLUMN_PRETTY_NAMES[plot_column])
    # plt.title(name.upper())

    default_label_map = {
        'torch_csr': 'Torch CSR',
        'torch_edge_list': 'Torch Edge List',
        'compiled_torch_edge_list': 'Torch Edge List (compiled)',
    }
    default_label_map.update(label_map or {})
    labels = [default_label_map.get(name, name) for name in df.columns]
    ax.spines[['right', 'top']].set_visible(False)
    # ax.legend(legend_handles[::-1], labels[::-1])

    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            if container.patches[0].get_facecolor()[-1] < 1.0:
                padding = 6
                label_type = 'center'
                replace_text = ''
            else:
                padding = 10
                label_type = 'edge'
                replace_text = 'OOM'
            # Set text size.
            # Make the labels appear  on top z.
            ax.bar_label(container, fmt=lambda x: f"{x:.2f}".lstrip('0') if x > 0 else replace_text,
                         padding=padding, fontsize=7, zorder=10, rotation=90,
                         label_type=label_type)
    # ax.relim()
    # # update ax.viewLim using the new dataLim
    # ax.autoscale_view()
    bars = ax.patches
    patterns = ('\\\\\\\\\\', '/////', '|||||', '....', 'xxxx', '++++')
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        if bar.get_facecolor()[-1] == 1.0:
            bar.set_hatch(hatch)

    if legend_outside:
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(labels, loc='upper left')

    plt.tight_layout()
    # put today's date in the filename
    clean_name = name.replace(',', '').replace('+', '').replace('  ', ' ').replace(':', '')

    if skip_timestamp:
        path = PLOT_FOLDER / 'thesis' / f'{clean_name}.pdf'
    else:
        path = PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} {clean_name}.pdf'

    plt.savefig(path, bbox_inches='tight')

    plt.title(name.upper())
    plt.tight_layout()
    plt.show()
