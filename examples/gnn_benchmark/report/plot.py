import pandas as pd
from matplotlib import pyplot as plt


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
    return df, std_df


def make_plot(full_df, name, label_map=None, bwd_df=None):
    df, std_df = prep_df(full_df)
    print(df)
    colors = get_colors(df.columns)
    # labels = ['DaCeML CSR CuSPARSE', 'PyG CSR', 'PyG adjacency list']
    # labels = ['DaCeML CSR CuSPARSE nomalloc', 'DaCeML CSR CuSPARSE',  'PyG CSR', 'PyG adjacency list']
    # labels = ['DaCeML', 'DaCeML + optimizations', 'PyG adjacency list', 'PyG CSR']
    if bwd_df is None:
        ax = df.plot(figsize=(6, 6), kind='barh', ylabel='Runtime [ms]', xlabel='Hidden size', color=colors,
                     xerr=std_df, label='Forward')
    else:
        bwd_df, bwd_std_df = prep_df(bwd_df)
        ax = bwd_df.plot(figsize=(6, 6),
                         kind='barh',
                         color=colors,
                         xerr=bwd_std_df,
                         label='Backward')
        df.plot(kind='barh',
                ylabel='Runtime [ms]',
                xlabel='Hidden size',
                color='white',
                alpha=0.3,
                xerr=std_df,
                ax=ax,
                label='Forward')

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='--')
    if bwd_df is not None and bwd_std_df is not None:
        ax.set_xlim(xmax=max((bwd_df + bwd_std_df).max().max() * 1.09, 0.9))
    else:
        ax.set_xlim(xmax=max((df + std_df).max().max() * 1.09, 0.9))
    ax.set_xlabel("Runtime [ms]")
    ax.set_ylabel("Hidden size")
    plt.title(name.upper())

    label_map = label_map or {}
    labels = [label_map.get(name, name) for name in df.columns]
    plt.legend(labels, loc='upper left' if name == 'gcn' else 'lower right')

    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.show()


def make_performance_plot(full_df, name):
    df = full_df.pivot(index='Size', columns='Name', values='Mean')
    std_df = full_df.pivot(index='Size', columns='Name', values='Stdev')
    flop_base = 13264 * 2 + 2 * 2704 * 1433
    print(df)
    flop = df.index * flop_base / (1000 * 1000 * 1000 * 1000)  # tflops
    # df['FLOP'] = flop
    print(df)
    df_flops = pd.DataFrame()
    print(df_flops)
    for col in df.columns:
        df_flops[col] = flop / (df[col] / 1000)
    print(df_flops)

    colors = ['mediumseagreen', 'indianred', 'lightcoral']

    ax = df_flops.plot(figsize=(5, 5), kind='barh', ylabel='Runtime [ms]', xlabel='Hidden size', color=colors)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='lightgray', linestyle='--')
    ax.set_xlabel("Performance [TFLOP / s]")
    ax.set_ylabel("Hidden size")
    peak_v100 = 15.7  # Page 5: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    ax.plot([peak_v100, peak_v100], [-10, 10], color='black', linestyle='--', linewidth=1)
    ax.text(peak_v100 * 0.95, 1, f'V100 peak: {peak_v100}', rotation=90, verticalalignment='center',
            horizontalalignment='left', fontsize=12)
    ax.set_xlim(xmax=peak_v100 * 1.1)
    plt.title(name.upper())
    # plt.legend(labels, loc='upper left' if name == 'gcn' else 'lower right')
    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(f'{name}-performance.pdf', bbox_inches='tight')
    plt.show()


def main():
    # tag = "11-04-gcn-csr-cora"
    # plot_block_sizes(tag)
    # plot_adapt_matmul_order()

    plot_backward("11-04-gcn-csr-cora-no-input-grad", model='GCN')
    plot_backward("11-04-gcn-single-csr-cora-no-input-grad", model='GCN Single Layer')


def plot_backward(tag, model):
    df = pd.read_csv(tag + '.csv')
    labels = {
        'dace_autoopt_persistent_mem_csr': 'DaCe CSR',
        'torch_csr': 'PyG CSR',
        'torch_edge_list': 'PyG edge list',
    }
    make_plot(df, "Forward pass", labels)
    bwd_df = pd.read_csv(tag + '-bwd.csv')
    make_plot(df, f"{model} Backward + forward pass", labels, bwd_df=bwd_df)


def plot_adapt_matmul_order():
    dfs = []
    for name in ['05-05-gcn-csr-cora-64-8-1.csv', '05-05-gcn-csr-adapt-cora-64-8-1.csv']:
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
    for name in ['05-05-gcn-csr-cora-64-8-1-bwd.csv', '05-05-gcn-csr-adapt-cora-64-8-1-bwd.csv']:
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
    make_plot(df, "Block size comparison (forward)", label_map=block_size_labels)
    dfs = []
    for sz in ['', '-64-8-1', '-512-1-1', '-128-8-1']:
        df_temp = pd.read_csv(tag + sz + '-bwd.csv')
        dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
        df_temp['Name'][dace_rows] = df_temp['Name'][dace_rows] + sz
        dfs.append(df_temp)
    df = pd.concat(dfs)
    make_plot(df, "Block size comparison (fwd + bwd)", label_map=block_size_labels)


if __name__ == '__main__':
    main()
