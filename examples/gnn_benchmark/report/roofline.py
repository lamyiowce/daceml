from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


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

    mem = 4 * (
            4 * num_nodes * df.index + num_nodes * num_nodes + 4 * df.index * num_edges + df.index) / 1000 / 1000 / 1000
    op_int = np.zeros((len(df.index), len(df.columns)))
    for i, col in enumerate(df_flops.columns):
        op_int[:, i] = df_flops[col] * 1000 / mem  # FLOPS / B

    df_op_int = pd.DataFrame(op_int, index=df.index, columns=df.columns)
    print(op_int)
    print("FLOPS", df_flops)
    print("Op INT", df_op_int)
    return df, df_flops, df_op_int


def make_roofline_plot(df, name):
    """Make a roofline plot for V100-PCIE-16GB."""
    _, df_flops, df_op_int = get_flops_df(df, name)
    df_flops = df_flops * 1000  # GFLOPS
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
    plt.text(1, peak_v100 * 1.1, f'Theoretical FP32: {peak_v100 / 1000} TFLOP/s', rotation=0,
             verticalalignment='bottom',
             horizontalalignment='left', fontsize=12)

    # Plot the memory bound.
    mem_bound = 900  # gb / s
    plt.plot([1 / mem_bound, peak_v100 / mem_bound], [1, peak_v100], color='black')
    plt.text(1, mem_bound * 1.1, f'HBM bandwidth: {mem_bound} GB/s', rotation=32,
             rotation_mode='anchor',
             fontsize=12)

    for col in df_flops.columns:
        plt.plot(df_op_int[col], df_flops[col], label=col)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}-roofline.pdf')
    plt.show()


def main():
    cora_single_df = pd.read_csv('gcn-single-cora.csv')
    make_roofline_plot(cora_single_df, name='gcn-single-cora',)
    # make_roofline_plot(arxiv_single_df, name='gcn-single-ogbn-arxiv',)


if __name__ == '__main__':
    main()
