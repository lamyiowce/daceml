from collections import OrderedDict
from io import StringIO
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from examples.gnn_benchmark.report import runtime_breakdown_data
from examples.gnn_benchmark.report.plot_common import PLOT_FOLDER, get_colors


def convert_duration_to_ms(duration_str: str) -> float:
    times = 0.001 if 'μs' in duration_str else 1
    amount = duration_str.split(' ')[0]
    return float(amount.replace(',', '.')) * times

total_fwd_ms = {
    'dace_csr': 3.91,
    'dace_coo': 3.77,
    'torch_edge_list_compiled': 3.06,
    'torch_csr': 5.64,
    'torch_edge_list': 9.52,
}

total_bwd_ms = {
    'dace_csr': 17.13,
    'dace_coo': 17.00,
    'torch_edge_list_compiled': 13.91,
    'torch_csr': 16.20,
    'torch_edge_list': 25.46,
}

def get_times(df: pd.DataFrame, benchmarked_time_ms: float, name=None) -> Dict[str, float]:
    names = df['Name']
    result = OrderedDict()
    total = 0
    other_filter = pd.Series([True] * len(names))
    for pretty_name, substr in [('Loss time', 'nll'),  ('GeMM time', 'gemm'), ('SpMM time', 'cusparse'), ('SpMM time', 'spmm_kernel')]:
        filter = names.str.contains(substr)
        other_filter = other_filter & ~filter
        if any(filter):
            print(f"{pretty_name} for {name}")
            print(df[filter])
        duration_sum = df[filter]['Duration'].sum()
        result[pretty_name] = result.get(pretty_name, 0) + duration_sum
        total += duration_sum

    print(f"Other for {name}")
    print(df[other_filter]['Name'])

    profile_time = df['Duration'].sum()
    profile_overhead = profile_time - benchmarked_time_ms
    other = max(benchmarked_time_ms - total, 0)
    # assert profile_overhead > 0
    if profile_overhead < 0:
        print(f"negative profile overhead: {profile_overhead}")
        profile_overhead = 0.
    result['Other'] = other
    result['Profile overhead'] = profile_overhead
    return result


colors = {
    'GeMM time': '#1f77b4',
    'SpMM time': '#ff7f0e',
    'Loss time': '#2ca02c',
    'Other': '#d62728',
    'Profile overhead': '#9467bd',
}

# For OGBN-Arxiv, hidden size 128, GCN.

# Profiling ovehead is the difference between (profiler-free) total runtime and the runtime reported in the profiling run.

def plot_df(df, ax, colors, label, title):
    """ Plot a dataframe as a horizontal bar plot. Stack the columns. """

    # df = df.drop(columns=['profiling total runtime', 'total runtime'])
    df.plot(figsize=(8, 8), kind='barh', xlabel='Runtime [ms]',
            color=colors, ax=ax, label=label, stacked=True)
    ax.set_title(title)
    for container in ax.containers:
        if hasattr(container, 'patches'):
            txt = ax.bar_label(container, fmt="%.2f", label_type='center',
                               color='black')
            for t in txt:
                if t._text !='0.00':
                    t.set_fontsize(10)
                else:
                    t.set_fontsize(0)
                # t.set_backgroundcolor('black')

        # Add grid lines.
    # ax.grid(axis='x', color='lightgray', linestyle='--')
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='lightgray', linestyle='--')
    # Put the legened outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # Make the legend fit in frame
    plt.tight_layout()


def prepare_data():
    fwd_data = {}
    for idx, data_str in runtime_breakdown_data.fwd_strs.items():
        all_fns_df = pd.read_csv(StringIO(data_str), sep='\t', usecols=['Name', 'Duration'],
                               converters={'Duration': convert_duration_to_ms})
        times = get_times(all_fns_df, total_fwd_ms[idx], name=idx)
        fwd_data[idx] = times

    bwd_data = {}
    for idx, data_str in runtime_breakdown_data.bwd_strs.items():
        all_fns_df = pd.read_csv(StringIO(data_str), sep='\t', usecols=['Name', 'Duration'],
                               converters={'Duration': convert_duration_to_ms})
        times = get_times(all_fns_df, total_bwd_ms[idx], name=idx)
        bwd_data[idx] = times

    fwd_df = pd.DataFrame.from_dict(fwd_data, orient='index')
    bwd_df = pd.DataFrame.from_dict(bwd_data, orient='index')
    return fwd_df, bwd_df


def plot():
    fwd_df, bwd_df = prepare_data()

    fig, axe = plt.subplots(figsize=(12, 8), nrows=2, ncols=1)

    for df, label, ax in zip([fwd_df, bwd_df], ['Forward', 'Fwd + Bwd'], axe):
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
