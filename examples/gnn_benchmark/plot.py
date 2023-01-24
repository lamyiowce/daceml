import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import colors


def make_plot(full_df, name):
    # fig = plt.figure(figsize=(5, 4))
    df = full_df.pivot(index='Size', columns='Name', values='Mean')
    std_df = full_df.pivot(index='Size', columns='Name', values='Stdev')
    print(df)
    colors = ['mediumseagreen', 'mediumaquamarine', 'indianred', 'lightcoral']
    labels = ['DaCeML', 'DaCeML + optimizations', 'PyG dense', 'PyG sparse']
    ax = df.plot(figsize=(5, 5), kind='barh', ylabel='Runtime [ms]', xlabel='Hidden size', color=colors, xerr=std_df)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='lightgray', linestyle='--')
    ax.set_xlim(xmax=max((df + std_df).max().max() * 1.09, 0.9))
    ax.set_xlabel("Runtime [ms]")
    plt.legend(labels, loc='upper left' if name == 'gcn' else 'lower right')
    plt.xticks(rotation=0)

    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt="%.2f")

    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.show()


def main():
    df = pd.read_csv('out.csv')
    make_plot(df[df.Model == 'gcn'], 'gcn')
    make_plot(df[df.Model == 'gat'], 'gat')


if __name__ == '__main__':
    main()