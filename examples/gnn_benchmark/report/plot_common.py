from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

DATA_FOLDER = Path(__file__).parent / 'data'

DEFAULT_LABEL_MAP = {
    'torch_csr': 'Torch CSR',
    'torch_edge_list': 'Torch Edge List',
    'torch_dgnn': 'Torch DGNN-GAT',
    'dace_csr': 'Dace CSR',
    'dace_csr_adapt': 'Dace CSR (adapt MM order)',
    'dace_coo': 'Dace COO',
    'dace_coo_adapt': 'Dace COO (adapt MM order)',
    'dace_csc': 'Dace CSC',
    'dace_csc_adapt': 'Dace CSC (adapt MM order)',
    'dace_csr_coo_adapt': 'Dace CSR/COO adapt, CSR 50%',
    'dace_csr_coo_adapt-0.99': 'Dace CSR/COO adapt, CSR 99%',
    'dace_csr_coo': 'Dace CSR/COO, CSR 50%',
    'dace_coo_cached': 'Dace COO (cached)',
    'dace_csc_cached': 'Dace CSC (cached)',
}

DEFAULT_LABEL_MAP.update(
    {f'dace_csr_coo_adapt-0.{i}': f'Dace CSR/COO adapt, CSR {i}0%' for i in
     range(3, 9)})

DEFAULT_LABEL_MAP.update(
    {f'{key}_compiled': f'{name} (compiled)' for key, name in
     DEFAULT_LABEL_MAP.items() if
     'torch' in key})

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
              'lightskyblue']
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


def prep_df(full_df):
    full_df = full_df.drop_duplicates(subset=['Size', 'Model', 'Name'])
    df = full_df.pivot(index='Size', columns='Name', values='Median')
    std_df = full_df.pivot(index='Size', columns='Name', values='Stdev')
    print(df)
    sorted_cols = sorted(df.columns,
                         key=lambda x: ('torch' in x, 'edge_list' in x))
    df = df.reindex(sorted_cols, axis=1)
    std_df = std_df.reindex(sorted_cols, axis=1)
    print(df)
    return df, std_df


def make_plot(full_df, name, label_map=None, bwd_df=None, legend_outside=False):
    df, std_df = prep_df(full_df)
    colors = get_colors(df.columns)
    bar_width = 0.75
    figsize = (6, 2 + len(df) * 1.2)
    if bwd_df is None:
        ax = df.plot(figsize=figsize, kind='barh', ylabel='Runtime [ms]',
                     xlabel='Hidden size', color=colors,
                     xerr=std_df, label='Forward', width=bar_width)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    else:
        bwd_df, bwd_std_df = prep_df(bwd_df)
        if len(bwd_df.columns) != len(df.columns):
            print('Warning: bwd_df and df have different lengths')
            print('Differing columns: ', set(bwd_df.columns) ^ set(df.columns))
        ax = bwd_df.plot(figsize=figsize,
                         kind='barh',
                         color=colors,
                         xerr=bwd_std_df,
                         label='Backward',
                         width=bar_width)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        df.plot(kind='barh',
                ylabel='Runtime [ms]',
                xlabel='Hidden size',
                color='white',
                alpha=0.3,
                xerr=std_df,
                ax=ax,
                label='Forward',
                width=bar_width)

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='--')
    if bwd_df is not None and bwd_std_df is not None:
        ax.set_xlim(xmax=max((bwd_df + bwd_std_df).max().max() * 1.09, 0.9))
    else:
        ax.set_xlim(xmax=max((df + std_df).max().max() * 1.09, 0.9))
    ax.set_xlabel("Runtime [ms]")
    ax.set_ylabel("Hidden size")
    plt.title(name.upper())

    default_label_map = {
        'torch_csr': 'Torch CSR',
        'torch_edge_list': 'Torch Edge List',
        'compiled_torch_edge_list': 'Torch Edge List (compiled)',
    }
    default_label_map.update(label_map or {})
    labels = [default_label_map.get(name, name) for name in df.columns]

    if legend_outside:
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(labels, loc='upper left' if name == 'gcn' else 'lower right')

    ax.legend(legend_handles[::-1], labels[::-1])

    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    # put today's date in the filename
    plt.savefig(
        PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} {name}.pdf',
        bbox_inches='tight')
    plt.show()
