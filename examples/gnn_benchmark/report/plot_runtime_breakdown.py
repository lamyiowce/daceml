import pandas as pd
from matplotlib import pyplot as plt

from examples.gnn_benchmark.report.plot import get_colors, PLOT_FOLDER

# Name,Model,Size,Min,Mean,Median,Stdev,Max
#
# torch_csr,gcn,128,5.625712890625,5.6350946044921875,5.630730285644532,0.016810760532441182,5.682094116210937
# torch_edge_list,gcn,128,9.503252563476563,9.504870544433594,9.504906311035157,0.0007733327155625472,9.506058349609376
# dace_csr,gcn,128,4.61275146484375,4.615893951416016,4.614210510253907,0.0064787205414138975,4.634234619140625
# dace_coo,gcn,128,4.58703857421875,4.5876868591308595,4.587581329345703,0.0005580447716394948,4.58893310546875

colors= {
    'GEMM time': '#1f77b4',
    'SpMM time': '#ff7f0e',
    'Loss time': '#2ca02c',
    'Other': '#d62728',
    'Profile overhead': '#9467bd',
}

# For OGBN-Arxiv, hidden size 128, GCN.
columns = ['Name', 'total runtime', 'GEMM time', 'SpMM time',
           'profiling total runtime']

# Profiling ovehead is the difference between (profiler-free) total runtime and the runtime reported in the profiling run.
data_fwd = [
    ('DaCe CSR', 4.614210510253907, 0.445805 + 0.276,
     0.01 + 1.459 + 0.01 + 0.555, 4.682),
    ('DaCe COO', 4.587581329345703, 0.444 + 0.278, 1.411 + 0.507, 4.6098),
    ('PyG CSR', 5.630730285644532, 0.482620 + 0.304541, 2.904 + 1.807, 6.032),
    # ('PyG Edge List', 9.504870544433594),
]

# torch_csr,gcn,128,16.18278350830078,16.20463592529297,16.20316162109375,0.018231768910547015,16.224972534179688
# torch_edge_list,gcn,128,25.303756713867188,25.317969970703125,25.31788787841797,0.012356765860416223,25.336370849609374
# dace_csr,gcn,128,17.86112060546875,17.866076354980468,17.867263793945312,0.004927424999086141,17.872589111328125
# dace_coo,gcn,128,17.82097930908203,17.84121368408203,17.843968200683594,0.012344169586933663,17.853797912597656

columns_bwd = ['Name', 'total runtime', 'Loss time', 'GEMM time', 'SpMM time',
               'profiling total runtime']

data_bwd = [
    ('DaCe CSR', 17.867263793945312, 6.3, 0.442 + 0.282 + 0.329 + 0.184 + 0.423,
     0.01 + 1.451 + 0.565 + 0.01 + 0.099 + 1.521 + 0.184 + 0.110 + 0.747 + 0.110 + 1.496,
     18.146),
    ('DaCe COO', 17.843968200683594, 6.3, 0.446 + 0.282 + 0.324 + 0.181 + 0.421,
     1.443 + 0.504 + 0.099 + 1.43 + 0.1 + 0.943 + 0.98 + 1.420, 17.9589),
    ('PyG CSR', 16.20316162, 6.3, 0.437 + 0.281 + 0.323 + 0.185 + 0.428,
     2.75 + 1.667 + 0.8 + 1.598, 16.4029),
    # ('PyG Edge List', 25.31788787841797, 0, 0, 25.4776),
]


def plot_df(df, ax, colors, label, title):
    """ Plot a dataframe as a horizontal bar plot. Stack the columns. """

    df['Other'] = df['total runtime'] - df['GEMM time'] - df['SpMM time']
    if 'Loss time' in df.columns:
        df['Other'] -= df['Loss time']
    df['Profile overhead'] = df['profiling total runtime'] - df['total runtime']
    df = df.drop(columns=['profiling total runtime', 'total runtime'])
    df.set_index(['Name'], inplace=True)
    df.plot(figsize=(8, 8), kind='barh', xlabel='Runtime [ms]',
            color=colors, ax=ax, label=label, stacked=True)
    ax.set_title(title)
    for container in ax.containers:
        if hasattr(container, 'patches'):
            txt = ax.bar_label(container, fmt="%.2f", label_type='center',
                               color='black')
            for t in txt:
                t.set_fontsize(10)
                # t.set_backgroundcolor('black')

        # Add grid lines.
    # ax.grid(axis='x', color='lightgray', linestyle='--')
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='--')
    # Put the legened outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # Make the legend fit in frame
    plt.tight_layout()


def plot():
    fig, axe = plt.subplots(figsize=(12, 8), nrows=2, ncols=1)

    fwd_df = pd.DataFrame(data_fwd, columns=columns)
    bwd_df = pd.DataFrame(data_bwd, columns=columns_bwd)

    for df, label, ax in zip([fwd_df, bwd_df], ['Forward', 'Backward'], axe):
        plot_df(df, ax, colors, None,
                f'{label} pass runtime breakdown, V100 / OGBN-Arxiv / GCN 128 hidden size')

    # plt.tight_layout()
    # put today's date in the filename
    plt.savefig(
        PLOT_FOLDER / f'{pd.Timestamp.today().strftime("%m-%d")} runtime breakdown.pdf',
        bbox_inches='tight')
    plt.show()

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='--')


if __name__ == '__main__':
    plot()
