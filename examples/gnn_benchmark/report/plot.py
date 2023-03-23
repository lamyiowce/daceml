import pandas as pd
from matplotlib import pyplot as plt


def make_plot(full_df, name):
    # fig = plt.figure(figsize=(5, 4))
    df = full_df.pivot(index='Size', columns='Name', values='Mean')
    std_df = full_df.pivot(index='Size', columns='Name', values='Stdev')
    print(df)
    colors = ['mediumseagreen', 'indianred', 'lightcoral']
    labels = ['DaCeML CSR CuSPARSE', 'PyG CSR', 'PyG adjacency list']
    # labels = ['DaCeML CSR CuSPARSE nomalloc', 'DaCeML CSR CuSPARSE',  'PyG CSR', 'PyG adjacency list']
    # colors = ['mediumseagreen', 'mediumaquamarine', 'indianred', 'lightcoral']
    # labels = ['DaCeML', 'DaCeML + optimizations', 'PyG adjacency list', 'PyG CSR']
    ax = df.plot(figsize=(5, 5), kind='barh', ylabel='Runtime [ms]', xlabel='Hidden size', color=colors, xerr=std_df)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='lightgray', linestyle='--')
    ax.set_xlim(xmax=max((df + std_df).max().max() * 1.09, 0.9))
    ax.set_xlabel("Runtime [ms]")
    ax.set_ylabel("Hidden size")
    plt.title(name.upper())
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
    for dataset in ['cora', 'citeseer', 'pubmed']:
        tag = f'03-23-gcn-csr-{dataset}'
        df = pd.read_csv(tag + '.csv')
        make_plot(df, tag)
        make_performance_plot(df, tag)
    # make_plot(df[df.Model == 'gat'], 'gat')


if __name__ == '__main__':
    main()
