import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DATA_FOLDER = Path(__file__).parent / 'data'
PLOT_FOLDER = Path(__file__).parent / 'plots'

DEFAULT_LABEL_MAP = {
    'torch_csr': 'Torch CSR',
    'torch_edge_list': 'Torch Edge List',
    'dace_csr': 'Dace CSR',
    'dace_coo': 'Dace COO',
    'dace_csc': 'Dace CSC',
}

def get_colors(names: pd.Series):
    reds = ['indianred', 'lightcoral', 'darkred', 'red']
    greens = ['lightgreen', 'mediumseagreen', 'mediumaquamarine', 'darkgreen']
    # Get unique names. If 'torch' is in the name, use red, otherwise green.
    unique_names = names.unique()
    color_dict = {}
    for name in unique_names:
        color_dict[name] = reds.pop(0) if 'torch' in name else greens.pop(0)
    return names.map(color_dict)


def prep_df(full_df):
    full_df = full_df.drop_duplicates(subset=['Size', 'Model', 'Name'])
    df = full_df.pivot(index='Size', columns='Name', values='Mean')
    std_df = full_df.pivot(index='Size', columns='Name', values='Stdev')
    sorted_cols = sorted(df.columns, key=lambda x: 'torch' in x)
    df = df.reindex(sorted_cols, axis=1)
    std_df = std_df.reindex(sorted_cols, axis=1)
    return df, std_df


def make_plot(full_df, name, label_map=None, bwd_df=None):
    df, std_df = prep_df(full_df)
    print(df)
    colors = get_colors(df.columns)
    bar_width = 0.75
    if bwd_df is None:
        ax = df.plot(figsize=(6, 8), kind='barh', ylabel='Runtime [ms]',
                     xlabel='Hidden size', color=colors,
                     xerr=std_df, label='Forward', width=bar_width)
    else:
        bwd_df, bwd_std_df = prep_df(bwd_df)
        ax = bwd_df.plot(figsize=(6, 8),
                         kind='barh',
                         color=colors,
                         xerr=bwd_std_df,
                         label='Backward',
                         width=bar_width)
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
    plt.legend(labels, loc='upper left' if name == 'gcn' else 'lower right')

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


def make_performance_plot(full_df, name, labels=None, title=None):
    assert 'single' in name
    df, df_flops, _ = get_flops_df(full_df, name)

    colors = get_colors(df.columns)
    default_labels = DEFAULT_LABEL_MAP.copy()
    default_labels.update(labels or {})
    labels = default_labels

    ax = df_flops.plot(figsize=(5, 5), kind='barh', ylabel='Runtime [ms]',
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
    ax.text(peak_v100 * 0.95, 1, f'V100 peak: {peak_v100} TFLOP / s', rotation=90,
            verticalalignment='center',
            horizontalalignment='left', fontsize=12)
    ax.set_xlim(xmax=peak_v100 * 1.1)
    plt.title(title or name.upper())
    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} {name}-performance.pdf', bbox_inches='tight')
    plt.show()


def get_flops_df(full_df, name):
    df = full_df.pivot(index='Size', columns='Name', values='Mean')
    if name == 'gcn-single-ogbn-arxiv':
        num_nodes = 169343
        num_edges = 1166243
        num_features = 128
    elif name == 'gcn-single-cora':
        num_nodes = 2708
        num_edges = 10556
        num_features = 1433
    else:
        raise ValueError(f'Unknown name: {name}')
    # This just need to be multiplied by hidden_size:
    # flop = 2 * num_nodes * num_features * hidden_size + 2 * num_edges * hidden_size + num_nodes * hidden_size
    flop_base = 2 * num_nodes * num_features + 2 * num_edges + num_nodes
    print(df)
    flop = df.index * flop_base / (1000 * 1000 * 1000 * 1000)  # tflops
    print(df)
    df_flops = pd.DataFrame()
    print(df_flops)
    for col in df.columns:
        df_flops[col] = flop / (df[col] / 1000)
    print(df_flops)


    mem = 4 * (4*num_nodes * df.index + num_nodes * num_nodes + 4 * df.index * num_edges + df.index) / 1000 / 1000 / 1000
    op_int = np.zeros((len(df.index), len(df.columns)))
    for i, col in enumerate(df_flops.columns):
        op_int[:, i] = df_flops[col] * 1000 / mem # FLOPS / B

    df_op_int = pd.DataFrame(op_int, index=df.index, columns=df.columns)
    print(op_int)
    print("FLOPS", df_flops)
    print("Op INT", df_op_int)
    return df, df_flops, df_op_int


def make_roofline_plot(df, name):
    """Make a roofline plot for V100-PCIE-16GB."""
    _, df_flops, df_op_int = get_flops_df(df, name)
    df_flops = df_flops * 1000 # GFLOPS
    ax = plt.figure(figsize=(8, 5))
    # Set both axes to logarithmic scaling.
    plt.xscale('log')
    plt.yscale('log')

    # Set axis limits.
    # plt.xlim(1e-1, 1e2)
    plt.ylim(1e-1, 1e5)

    # Set axis labels.
    plt.xlabel('Operational Intensity [FLOP/Byte]')
    plt.ylabel('Performance [GFLOP/s]')

    # Plot the theoretical performance peak for V100.
    peak_v100 = 14 * 1000  # https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
    plt.plot([1e-1, 1e2], [peak_v100, peak_v100], color='black')
    plt.text(1, peak_v100 * 1.1, f'Theoretical FP32: {peak_v100/1000} TFLOP/s', rotation=0,
            verticalalignment='bottom',
            horizontalalignment='left', fontsize=12)

    # Plot the memory bound.
    mem_bound = 900 # gb / s
    plt.plot([1/mem_bound, peak_v100 / mem_bound], [1, peak_v100], color='black')
    plt.text(1, mem_bound * 1.1, f'HBM bandwidth: {mem_bound} GB/s', rotation=32, rotation_mode='anchor',
             fontsize=12)

    for col in df_flops.columns:
        plt.plot(df_op_int[col], df_flops[col], label=col)

    plt.legend()
    plt.show()



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
            df_temp.loc[dace_rows, 'Name'] = df_temp.loc[dace_rows, 'Name'].apply(
                name_fn) if name_fn else name
        dfs.append(df_temp)
        if backward:
            bwd_path = DATA_FOLDER / filename.replace('.csv', '-bwd.csv')
            df_temp = pd.read_csv(bwd_path, comment='#')
            if name_to_replace:
                dace_rows = df_temp['Name'].str.contains(name_to_replace)
                df_temp.loc[dace_rows, 'Name'] = df_temp.loc[dace_rows, 'Name'].apply(
                    name_fn) if name_fn else name
            bwd_dfs.append(df_temp)

    return pd.concat(dfs), pd.concat(bwd_dfs)


def main():
    # tag = "11-04-gcn-csr-cora"
    # plot_block_sizes(tag)
    # plot_adapt_matmul_order()

    # plot_backward("data/24-04-gcn-reduce-gpuauto-simplify", model='GCN')
    # plot_backward("data/24-04-gcn-single-reduce-gpuauto-simplify", model='GCN Single layer')

    arxiv_df, arxiv_bwd_df = read_many_dfs(
        filenames=['10.05.15.33-pyg-gcn-ogbn-arxiv-191680.csv',
                   '10.05.08.35-fix-contiguous-gcn-ogbn-arxiv-191411.csv',
                   '10.05.16.28-gcn-ogbn-arxiv-191708.csv']
    )
    plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv', plot_title="OGB Arxiv")

    cora_df, cora_bwd_df = read_many_dfs(
        filenames=['10.05.09.03-fix-contiguous-gcn-cora-191411.csv',
                   '10.05.15.40-pyg-gcn-cora-191680.csv',
                   '10.05.16.08-gcn-cora-191708.csv']
    )
    plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-ogbn-cora', plot_title="Cora")


    # cora_single_df, _ = read_many_dfs(
    #     filenames=['04.05.10.07-gcn_single_layer-cora-186177.csv',]
    # )
    # make_performance_plot(full_df=cora_single_df, name='gcn-single-cora', title="Single GCN layer performance, Cora")
    #
    # arxiv_single_df, _ = read_many_dfs(
    #     filenames=['04.05.10.25-gcn_single_layer-ogbn-arxiv-186177.csv',]
    # )
    # make_performance_plot(full_df=arxiv_single_df, name='gcn-single-ogbn-arxiv', title="Single GCN layer performance, OGB Arxiv")

    # make_roofline_plot(cora_single_df, name='gcn-single-cora',)
    # make_roofline_plot(arxiv_single_df, name='gcn-single-ogbn-arxiv',)
    # plot_midthesis_main_datasets()
    # plot_midthesis_additional_datasets()
    # plot_stream_comparison()


def plot_midthesis_additional_datasets():
    plot_backward('03.05.09.52-gcn-pubmed-185244', plot_title='GCN, Pubmed')
    plot_backward('03.05.10.26-gcn-flickr-185244', plot_title='GCN, Flickr')
    plot_backward('03.05.10.09-gcn-citeseer-185244', plot_title='GCN, Citeseer')


def plot_midthesis_main_datasets():
    arxiv_df, arxiv_bwd_df = read_many_dfs(
        filenames=['01.05.11.34-gcn-ogbn-arxiv-183528.csv',
                   '01.05.12.33-gcn-ogbn-arxiv-183567.csv',
                   '03.05.17.23-gcn-ogbn-arxiv-csc-185561.csv',
                   '03.05.18.24-gcn-ogbn-arxiv-alt-sizes-185598.csv',
                   '03.05.19.41-gcn-ogbn-arxiv-alt-sizes-185634.csv']
    )
    plot_backward(tag='arxiv', sizes=[16, 64, 256, 1024], plot_title='GCN, OGBN Arxiv', df=arxiv_df,
                  bwd_df=arxiv_bwd_df)
    plot_backward(tag='arxiv', plot_title='GCN, OGBN Arxiv', df=arxiv_df, bwd_df=arxiv_bwd_df)
    cora_df, cora_bwd_df = read_many_dfs(
        filenames=['01.05.11.17-gcn-cora-183528.csv',
                   '03.05.17.14-gcn-cora-csc-185561.csv',
                   '03.05.18.04-gcn-cora-alt-sizes-185598.csv'])
    plot_backward(tag="cora", sizes=[16, 64, 256, 1024], plot_title='GCN, Cora', df=cora_df, bwd_df=cora_bwd_df)
    plot_backward(tag="cora", plot_title='GCN, Cora', df=cora_df, bwd_df=cora_bwd_df)


def plot_stream_comparison():
    df, bwd_df = read_many_dfs(
        filenames=['26-04-gcn-csr-coo-cora-single-stream.csv',
                   '26-04-gcn-csr-coo-cora-many-streams.csv'],
        name_fns=[lambda s: s + "_single_stream",
                  lambda s: s + "_many_streams"],
        name_to_replace='dace_.*')
    labels = {
        "dace_autoopt_persistent_mem_coo_single_stream": "DaCe COO single stream",
        "dace_autoopt_persistent_mem_coo_many_streams": "DaCe COO many streams",
        "dace_autoopt_persistent_mem_csr_single_stream": "DaCe CSR single stream",
        "dace_autoopt_persistent_mem_csr_many_streams": "DaCe CSR many streams",
    }
    make_plot(df, f"GCN Backward + forward pass, Cora, V100", bwd_df=bwd_df,
              label_map=labels)


def plot_backward(tag, plot_title, labels=None, df=None, bwd_df=None, sizes=None):
    if df is None:
        df = pd.read_csv(DATA_FOLDER / (tag + '.csv'), comment='#')
    if sizes is not None:
        df = df[df['Size'].isin(sizes)]

    default_labels = DEFAULT_LABEL_MAP.copy()
    default_labels.update(labels or {})
    labels = default_labels
    make_plot(df, f"{plot_title}: forward pass", labels)
    if bwd_df is None:
        bwd_path = DATA_FOLDER / (tag + '-bwd.csv')
        if bwd_path.exists():
            bwd_df = pd.read_csv(bwd_path, comment='#')
        else:
            print(f"Could not find backward file {bwd_path}.")
    if bwd_df is not None:
        if sizes is not None:
            bwd_df = bwd_df[bwd_df['Size'].isin(sizes)]

        make_plot(df, f"{plot_title}:  Backward + forward pass", labels, bwd_df=bwd_df)


def plot_adapt_matmul_order():
    dfs = []
    for name in ['05-05-gcn-csr-cora-64-8-1.csv',
                 '05-05-gcn-csr-adapt-cora-64-8-1.csv']:
        df_temp = pd.read_csv(name)
        dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
        df_temp['Name'][dace_rows] = name
        dfs.append(df_temp)
    df = pd.concat(dfs)
    adapt_labels = {
        '05-05-gcn-csr-cora-64-8-1.csv': 'No adapt',
        '05-05-gcn-csr-adapt-cora-64-8-1.csv': 'Adapt matmul order',
    }
    make_plot(df, "Adapt matmul order vs no adapt (forward)", adapt_labels)
    dfs = []
    for name in ['05-05-gcn-csr-cora-64-8-1-bwd.csv',
                 '05-05-gcn-csr-adapt-cora-64-8-1-bwd.csv']:
        df_temp = pd.read_csv(name)
        dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
        df_temp['Name'][dace_rows] = name
        dfs.append(df_temp)
    df = pd.concat(dfs)
    make_plot(df, "Adapt matmul order vs no adapt (fwd+bwd)", adapt_labels)
    return adapt_labels


def plot_block_sizes(tag):
    dfs = []
    for sz in ['', '-64-8-1', '-512-1-1', '-128-8-1']:
        df_temp = pd.read_csv(tag + sz + '.csv')
        dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
        df_temp['Name'][dace_rows] = df_temp['Name'][dace_rows] + sz
        dfs.append(df_temp)
    df = pd.concat(dfs)
    block_size_labels = {
        'dace_autoopt_persistent_mem_csr': 'Block size 32,1,1',
        'dace_autoopt_persistent_mem_csr-64-8-1': 'Block size 64,8,1',
        'dace_autoopt_persistent_mem_csr-512-1-1': 'Block size 512,1,1',
        'dace_autoopt_persistent_mem_csr-128-8-1': 'Block size 128,8,1',
    }
    make_plot(df, "Block size comparison (forward)",
              label_map=block_size_labels)
    dfs = []
    for sz in ['', '-64-8-1', '-512-1-1', '-128-8-1']:
        df_temp = pd.read_csv(tag + sz + '-bwd.csv')
        dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
        df_temp['Name'][dace_rows] = df_temp['Name'][dace_rows] + sz
        dfs.append(df_temp)
    df = pd.concat(dfs)
    make_plot(df, "Block size comparison (fwd + bwd)",
              label_map=block_size_labels)


if __name__ == '__main__':
    main()
