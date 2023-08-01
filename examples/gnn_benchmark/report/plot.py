import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from examples.gnn_benchmark.report.plot_common import read_many_dfs, \
    DATA_FOLDER, DEFAULT_LABEL_MAP, make_plot


def main():
    # 23.07
    plot_gcn_schemes()

    # 18.07 Thesis
    plot_gcn_thesis()
    # plot_compare_cutoffs()

    # 06.07 plot GAT bwd
    plot_gat_bwd()

    # # 16.06 plot GAT forward single layer.
    # plot_gat_single_layer()
    #
    # # 15.06 plot GCN forward single layer.
    # plot_gcn_single_layer()
    #
    # # 06.06 Plot GAT fwd with multiple spmm kernels and permutations.
    # #
    # plot_gat_model()

    # # 06.06 Plot GCN after block size adaptation.
    # This is actually incorrect! some kernels are not run!!!!
    # arxiv_df, arxiv_bwd_df = read_many_dfs(
    #     filenames=['10.05.15.33-pyg-gcn-ogbn-arxiv-191680.csv',
    #                '11.05-pyg-arxiv-1024.csv',
    #                '05.06.15.02-gcn-ogbn-arxiv-203054.csv']
    # )
    # plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv',
    #               plot_title="GCN, OGB Arxiv")
    #
    # cora_df, cora_bwd_df = read_many_dfs(
    #     filenames=['05.06.14.22-gcn-cora-203054.csv',
    #                '10.05.15.40-pyg-gcn-cora-191680.csv', ]
    # )
    # plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-cora',
    #               plot_title="GCN, Cora")

    # cora_df, cora_bwd_df = read_many_dfs(
    #     filenames=['02.06.09.52-block-sizes-gcn-cora-202957.csv']
    # )
    # plot_block_sizes(cora_df, cora_bwd_df, name='Cora')
    # arxiv_df, arxiv_bwd_df = read_many_dfs(
    #     filenames=['02.06.10.44-block-sizes-gcn-ogbn-arxiv-202957.csv']
    # )
    # plot_block_sizes(arxiv_df, arxiv_bwd_df, name='Arxiv')

    # plot_block_sizes(tag)
    # plot_adapt_matmul_order()

    # plot_backward("data/24-04-gcn-reduce-gpuauto-simplify", model='GCN')
    # plot_backward("data/24-04-gcn-single-reduce-gpuauto-simplify", model='GCN Single layer')

    # plot_compare_cutoffs()

    # arxiv_df, arxiv_bwd_df = read_many_dfs(
    #     filenames=['10.05.15.33-pyg-gcn-ogbn-arxiv-191680.csv',
    #                '10.05.08.35-fix-contiguous-gcn-ogbn-arxiv-191411.csv',
    #                '10.05.16.28-gcn-ogbn-arxiv-191708.csv',
    #                '11.05-pyg-arxiv-1024.csv',
    #                '16.05.12.53-gcn-ogbn-arxiv-196721.csv',
    #                '16.05.13.59-gcn-csr_adapt-ogbn-arxiv-196780.csv',
    #                '23.05.14.02-gcn-ogbn-arxiv-202457.csv']
    # )
    # plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv', plot_title="OGB Arxiv",
    #               drop_names=['dace_csc', 'dace_coo', 'dace_csr'])

    # cora_df, cora_bwd_df = read_many_dfs(
    #     filenames=['10.05.09.03-fix-contiguous-gcn-cora-191411.csv',
    #                '10.05.15.40-pyg-gcn-cora-191680.csv',
    #                '10.05.16.08-gcn-cora-191708.csv',
    #                '16.05.12.40-gcn-cora-196721.csv',
    #                '16.05.13.45-gcn-csr_adapt-cora-196780.csv',
    #                '23.05.12.56-gcn-cora-202421.csv']
    # )
    # plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-ogbn-cora', plot_title="Cora",
    #               drop_names=['dace_csc', 'dace_coo', 'dace_csr'])

    #
    # gat_arxiv_df, gat_arxiv_bwd_df = read_many_dfs(['18.05.14.46-pyg-gat-ogbn-arxiv-198393.csv'], backward=True)
    # plot_backward(df=gat_arxiv_df, bwd_df=gat_arxiv_bwd_df, tag='', plot_title="GAT Arxiv")
    #
    # gatcoradf, gatcorabwd_df = read_many_dfs(['18.05.14.43-pyg-gat-cora-198393.csv'], backward=True)
    # plot_backward(df=gatcoradf, bwd_df=gatcorabwd_df, tag='', plot_title="GAT Cora")

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


def plot_gcn_schemes():
    data = {
        "No input grad": [
            '24.07.17.59-gcn_single_layer-ogbn-arxiv-221825.csv',
            '24.07.16.00-gcn_single_layer-ogbn-arxiv-221752.csv',
            '24.07.13.43-gcn_single_layer-ogbn-arxiv-221634.csv',
        ],
        "With input grad": [
            '24.07.17.14-gcn_single_layer-ogbn-arxiv-221824.csv',
            '24.07.15.57-gcn_single_layer-ogbn-arxiv-input-grad-221749.csv',
        ],
    }
    subsets = {
        'with caching': ['dace_csc', 'dace_csc_cached', 'dace_csc_adapt_cached'],
        'no caching': ['dace_csc', 'dace_csc_alt', 'dace_csc_adapt'],
    }

    labels = {
        'dace_csc': 'FWD: Transform-first, BWD: fused-propagate',
        'dace_csc_alt': 'FWD: Propagate-first, BWD: split-propagate',
        'dace_csc_adapt': 'Adaptive, no caching',
        'dace_csc_cached': 'FWD: Propagate-first, BWD: split-propagate, with caching',
        'dace_csc_adapt_cached': 'Adaptive with caching',
    }

    for name, datalist in data.items():
        if len(datalist) > 0:
            for subset_name, subset in subsets.items():
                df, bwd_df = read_many_dfs(filenames=datalist)
                plot_backward(df=df[df['Num Features'] == 128],
                              bwd_df=bwd_df[bwd_df['Num Features'] == 128],
                              tag='GCN ' + name, plot_column='Size',
                              plot_title=f"GCN Single Layer, {subset_name}, {name}",
                              include_only_names=subset,
                              labels=labels,
                              skip_timestamp=True,
                              xlabel='Output features size')

                # plot_backward(df=df[df['Size'] == 128],
                #               bwd_df=bwd_df[bwd_df['Size'] == 128], tag='GCN ' + name,
                #               plot_column='Num Features',
                #               plot_title=f"GCN Single Layer, {subset_name}, {name}",
                #               include_only_names=subset,
                #               labels=labels,
                #               skip_timestamp=True)


def plot_gcn_thesis():
    drop_names = ['torch_edge_list', 'dace_csc_coo_adapt_cached-0.9']
    data = {
        "OGB Arxiv": [
            '25.07.07.19-pyg-gcn-ogbn-arxiv-222314.csv',
            # '21.07.12.51-pyg-gcn-ogbn-arxiv-219071.csv',
            '24.07.21.59-gcn-ogbn-arxiv-221963.csv',
            # '21.07.10.55-pyg-gcn-ogbn-arxiv-219003.csv',
            # '25.07.11.49-gcn-ogbn-arxiv-222541.csv',
            '25.07.12.44-gcn-ogbn-arxiv-222570.csv',
        ],
        "Cora": [
            # '21.07.10.53-pyg-gcn-cora-219003.csv',
            '24.07.22.27-gcn-cora-221963.csv',
            '25.07.07.26-pyg-gcn-cora-222314.csv',
            '25.07.12.20-gcn-cora-222570.csv',
            # '21.07.12.50-pyg-gcn-cora-219071.csv',
            # '18.07.15.14-gcn-cora-216728.csv',
            # '20.07.13.15-gcn-cora-218300.csv',
            # '19.07.11.57-pyg-gcn-cora-217423.csv',
        ],
        "Citeseer": [
            '25.07.07.32-pyg-gcn-citeseer-222314.csv',
            '25.07.07.32-pyg-gcn-citeseer-222314.csv',
            '25.07.13.31-gcn-citeseer-222570.csv',
            '27.07.07.41-gcn-citeseer-224007.csv',
        ],
        "Pubmed": [
            '25.07.07.28-pyg-gcn-pubmed-222314.csv',
            '25.07.07.28-pyg-gcn-pubmed-222314.csv',
            '25.07.13.08-gcn-pubmed-222570.csv',
            '27.07.07.36-gcn-pubmed-224007.csv',
        ],
        "Flickr": [
            '25.07.07.18-gcn-flickr-222313.csv',
            '25.07.12.21-gcn-flickr-222571.csv',
            '25.07.07.35-pyg-gcn-flickr-222314.csv',
        ],
        "Reddit": [
            # '25.07.07.46-gcn-reddit-222313.csv',
            # '25.07.07.39-pyg-gcn-reddit-222314.csv',
            # '25.07.12.46-gcn-reddit-222571.csv',
        ]
    }

    speedup_log = open('speedup_log_gcn.txt', 'w')

    all_speedups = {'fwd': [], 'bwd': [], 'formats_fwd': [], 'formats_bwd': []}
    per_dataset_format_speedups_fwd = {}
    per_dataset_format_speedups_bwd = {}
    for name, datalist in data.items():
        if len(datalist) > 0:
            df, bwd_df = read_many_dfs(filenames=datalist)
            plot_backward(df=df, bwd_df=bwd_df, tag='GCN ' + name,
                          plot_title=f"GCN, {name}", drop_names=drop_names, skip_timestamp=True)

            dataset_speedups = {'fwd': [], 'bwd': [], 'formats_fwd': [], 'formats_bwd': []}
            for pass_name, data in [('fwd', df), ('bwd', bwd_df)]:
                torch_df = data[data['Name'].str.contains('torch')]
                dace_df = data[data['Name'].str.contains('dace')]

                for size in data['Size'].unique():
                    torch_time = torch_df[torch_df['Size'] == size]['Median'].min()
                    dace_time = dace_df[dace_df['Size'] == size]['Median'].min()
                    dace_max_time = dace_df[dace_df['Size'] == size]['Median'].max()
                    dataset_speedups[f'formats_{pass_name}'].append(dace_max_time / dace_time)
                    dataset_speedups[pass_name].append(torch_time / dace_time)
                    print(f"{pass_name}: {name}, {size}: {torch_time / dace_time}")
                    speedup_log.write(f"{pass_name}: {name}, {size}: {torch_time / dace_time}\n")

            # Compute geomean speedup for this dataset.
            for pass_name, speedups in dataset_speedups.items():
                print(f"{name} {pass_name} geomean: {stats.gmean(speedups)}")
                speedup_log.write(f"{name} {pass_name} geomean: {stats.gmean(speedups)}\n")
                print(f"{name} {pass_name} max: {max(speedups)}")
                speedup_log.write(f"{pass_name} max: {max(speedups)}\n")
                print(f"{name} {pass_name} min: {min(speedups)}")
                speedup_log.write(f"{pass_name} min: {min(speedups)}\n")

            per_dataset_format_speedups_fwd[name] = dataset_speedups['formats_fwd']
            per_dataset_format_speedups_bwd[name] = dataset_speedups['formats_bwd']
            all_speedups['fwd'] += dataset_speedups['fwd']
            all_speedups['bwd'] += dataset_speedups['bwd']
            all_speedups['formats_fwd'] += dataset_speedups['formats_fwd']
            all_speedups['formats_bwd'] += dataset_speedups['formats_bwd']

    # Compute geomean, max and min speedup.
    for pass_name, speedups in all_speedups.items():
        print(f"ALL DATASETS {pass_name} geomean: {stats.gmean(speedups)}")
        speedup_log.write(f"{pass_name} geomean: {stats.gmean(speedups)}\n")
        print(f"ALL DATASETS {pass_name} max: {max(speedups)}")
        speedup_log.write(f"{pass_name} max: {max(speedups)}\n")
        print(f"ALL DATASETS {pass_name} min: {min(speedups)}")
        speedup_log.write(f"{pass_name} min: {min(speedups)}\n")

    dataset_to_size = {
        'Cora': 2708,
        'Citeseer': 3327,
        'Pubmed': 19717,
        'Flickr': 89250,
        'Reddit': 232965,
        'OGB Arxiv': 169343,
    }
    dataset_to_num_edges = {
        'Cora': 5278,
        'Citeseer': 4732,
        'Pubmed': 44338,
        'Flickr': 899756,
        'Reddit': 114615892,
        'OGB Arxiv': 1166243,
    }
    for pass_name, speedups in [('fwd', per_dataset_format_speedups_fwd), ('bwd', per_dataset_format_speedups_bwd)]:
        df = pd.DataFrame(speedups)
        df.rename(columns=dataset_to_num_edges, inplace=True)
        df.plot()
        plt.title(f"GCN {pass_name} speedup per dataset")
        plt.show()




    # for name, datalist in data.items():
    #     if len(datalist) > 0:
    #         df, bwd_df = read_many_dfs(filenames=datalist)
    #         plot_backward(df=df, bwd_df=bwd_df, tag='GCN ' + name + ' short',
    #                       plot_title=f"GCN, {name}", drop_names=drop_names,
    #                       sizes=[8, 32, 128, 512])


def plot_gat_bwd():
    drop_torch_names = ['torch_edge_list', 'torch_dgnn', 'torch_dgnn_compiled', 'torch_edge_list_compiled', 'torch_csr']
    drop_dace_names = ['torch_edge_list', 'dace_coo', 'dace_coo_cached_feat_and_alpha', 'dace_coo_cached:coo_cached_feat_only']
    col_order = ['dace_coo_cached', 'dace_coo_cached_feat_and_alpha', 'dace_coo_cached:coo_cached_feat_only', 'dace_coo']
    labels = {
        'dace_coo_cached': 'Full caching',
        'dace_coo_cached_feat_and_alpha': 'Cache features and node attention',
        'dace_coo_cached:coo_cached_feat_only': 'Cache only features',
        'dace_coo': 'No caching',
    }

    sizes = None #[8, 16, 32, 64, 128]

    data = {
        "OGB Arxiv": [
            '27.07.12.18-pyg-gat-ogbn-arxiv-224204.csv',
            '27.07.12.45-pyg-gat-ogbn-arxiv-224205.csv',
            '27.07.12.18-pyg-gat-ogbn-arxiv-224204.csv',
            '31.07.16.48-gat-ogbn-arxiv-227723.csv',
        ],
        "Cora": [
            '27.07.12.23-pyg-gat-cora-224204.csv',
            '27.07.12.52-pyg-gat-cora-224205.csv',
            '31.07.15.58-gat-cora-227723.csv',
        ],
        "Citeseer": [
            '27.07.13.00-pyg-gat-citeseer-224205.csv',
            '31.07.15.59-gat-citeseer-227724.csv',
        ],
        "Pubmed": [
            '27.07.13.45-pyg-gat-pubmed-224242.csv',
            '27.07.13.41-pyg-gat-pubmed-224239.csv',
            '27.07.12.26-pyg-gat-pubmed-224204.csv',
            '27.07.12.55-pyg-gat-pubmed-224205.csv',
            '31.07.15.56-gat-pubmed-227722.csv',
        ],
        "Flickr": [
            '27.07.13.46-pyg-gat-flickr-224242.csv',
            '27.07.13.04-pyg-gat-flickr-224205.csv',
            '27.07.12.33-pyg-gat-flickr-224204.csv',
            '31.07.16.47-gat-flickr-227722.csv',
        ],
        # "Reddit": [
        #     '27.07.13.10-pyg-gat-reddit-224205.csv',
        #
        # ]
    }

    for name, datalist in data.items():
        if len(datalist) > 0:
            df, bwd_df = read_many_dfs(filenames=datalist)
            plot_backward(df=df, bwd_df=bwd_df, tag='GAT ' + name, filter_y=sizes, col_order=col_order,
                          plot_title=f"GAT COMPARISON, {name}", drop_names=drop_torch_names, skip_timestamp=True, labels=labels)
            plot_backward(df=df, bwd_df=bwd_df, tag='GAT ' + name, filter_y=sizes,
                          plot_title=f"GAT BASELINES, {name}", drop_names=drop_dace_names, skip_timestamp=True, labels=labels)



def plot_compare_cutoffs():
    data = {
        'Cora': [
            '19.07.14.44-gcn-cora-csccoo-compare-217536.csv',
            '19.07.14.44-gcn-cora-csrcoo-compare-217535.csv',
            '19.07.11.57-pyg-gcn-cora-217423.csv',
        ],
        'Arxiv': [
            '19.07.15.11-gcn-ogbn-arxiv-csrcoo-compare-217535.csv',
            '19.07.15.11-gcn-ogbn-arxiv-csccoo-compare-217536.csv',
            '19.07.12.02-pyg-gcn-ogbn-arxiv-217423.csv',
        ]
    }

    for name, datalist in data.items():
        if len(datalist) > 0:
            df, bwd_df = read_many_dfs(filenames=datalist)
            plot_backward(df=df, bwd_df=bwd_df, tag='GCN ' + name,
                          plot_title=f"GCN, {name}", filter_y=[8, 256], legend_outside=True)

    # Old
    # arxiv_df, arxiv_bwd_df = read_many_dfs(
    #     filenames=['10.05.15.33-pyg-gcn-ogbn-arxiv-191680.csv',
    #                '11.05-pyg-arxiv-1024.csv',
    #                '16.05.12.53-gcn-ogbn-arxiv-196721.csv',
    #                '16.05.13.59-gcn-csr_adapt-ogbn-arxiv-196780.csv',
    #                '23.05.13.27-gcn-ogbn-arxiv-202455.csv']
    # )
    # plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv',
    #               plot_title="OGB Arxiv, cutoff comparison",
    #               drop_names=['torch_edge_list', 'torch_csr'], sizes=[16, 256],
    #               legend_outside=True)


def plot_gat_single_layer():
    arxiv_df, arxiv_bwd_df = read_many_dfs(
        filenames=[
            '16.06.12.42-gat_single_layer-ogbn-arxiv-203724.csv',
            '16.06.13.06-pyg-gat_single_layer-ogbn-arxiv-203727.csv',
        ],
        backward=False
    )
    plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv',
                  plot_title="GAT Single Layer, OGB Arxiv", filter_y=[8, 16, 32, 64, 128, 256],
                  drop_names=['torch_edge_list'])
    cora_df, cora_bwd_df = read_many_dfs(
        filenames=[
            '16.06.12.32-gat_single_layer-cora-203724.csv',
            '16.06.12.58-pyg-gat_single_layer-cora-203727.csv',
        ],
        backward=False
    )
    plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-cora',
                  plot_title="GAT Single Layer, Cora", filter_y=[8, 16, 32, 64, 128, 256])


def plot_gat_model():
    arxiv_df, arxiv_bwd_df = read_many_dfs(
        filenames=[
            '18.05.14.46-pyg-gat-ogbn-arxiv-198393.csv',
            '06.06.16.48-pyg-gat-ogbn-arxiv-203173.csv',
            '08.06.12.40-gat-ogbn-arxiv-203320.csv',
            '09.06.12.55-gat-ogbn-arxiv-203470.csv',
        ],
        backward=False
    )
    plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv',
                  plot_title="GAT, OGB Arxiv", filter_y=[8, 16, 32, 64, 128, 256])
    cora_df, cora_bwd_df = read_many_dfs(
        filenames=['18.05.14.43-pyg-gat-cora-198393.csv',
                   '18.05.14.59-pyg-gat-cora-198400.csv',
                   '06.06.16.41-pyg-gat-cora-203173.csv',
                   '08.06.12.33-gat-cora-203320.csv',
                   '09.06.12.50-gat-cora-203470.csv'
                   ],
        backward=False
    )
    plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-cora',
                  plot_title="GAT, Cora", filter_y=[8, 16, 32, 64, 128, 256])


def plot_gcn_single_layer():
    arxiv_df, arxiv_bwd_df = read_many_dfs(
        filenames=[
            '15.06.12.59-gcn_single_layer-ogbn-arxiv-203691.csv',
            '15.06.13.39-pyg-gcn_single_layer-ogbn-arxiv-203692.csv',
            '16.06.11.00-gcn_single_layer-ogbn-arxiv-203723.csv',
            '16.06.12.41-pyg-gcn_single_layer-ogbn-arxiv-203725.csv',
        ],
        backward=True
    )
    plot_backward(df=arxiv_df, bwd_df=arxiv_bwd_df, tag='gcn-ogbn-arxiv',
                  plot_title="GCN Single Layer, OGB Arxiv", filter_y=[8, 16, 32, 64, 128, 256],
                  drop_names=['torch_edge_list', 'torch_csr'])
    cora_df, cora_bwd_df = read_many_dfs(
        filenames=[
            '15.06.11.23-gcn_single_layer-cora-203685.csv',
            '15.06.13.34-pyg-gcn_single_layer-cora-203692.csv',
            '16.06.12.39-pyg-gcn_single_layer-cora-203725.csv',
        ],
        backward=True
    )
    plot_backward(df=cora_df, bwd_df=cora_bwd_df, tag='gcn-cora',
                  plot_title="GCN Single Layer, Cora", filter_y=[8, 16, 32, 64, 128, 256])


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
    plot_backward(tag='arxiv', filter_y=[16, 64, 256, 1024],
                  plot_title='GCN, OGBN Arxiv', df=arxiv_df,
                  bwd_df=arxiv_bwd_df)
    plot_backward(tag='arxiv', plot_title='GCN, OGBN Arxiv', df=arxiv_df,
                  bwd_df=arxiv_bwd_df)
    cora_df, cora_bwd_df = read_many_dfs(
        filenames=['01.05.11.17-gcn-cora-183528.csv',
                   '03.05.17.14-gcn-cora-csc-185561.csv',
                   '03.05.18.04-gcn-cora-alt-sizes-185598.csv'])
    plot_backward(tag="cora", filter_y=[16, 64, 256, 1024], plot_title='GCN, Cora',
                  df=cora_df,
                  bwd_df=cora_bwd_df)
    plot_backward(tag="cora", plot_title='GCN, Cora', df=cora_df,
                  bwd_df=cora_bwd_df)


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
    make_plot(df, f"GCN Backward + forward, Cora, V100", bwd_df=bwd_df,
              label_map=labels)


def plot_backward(tag, plot_title, plot_column='Size', labels=None, df=None, bwd_df=None,
                  filter_y=None, drop_names=None, include_only_names=None, color_map=None,
                  legend_outside=False, skip_timestamp=False, xlabel=None, col_order=None):
    if df is None:
        df = pd.read_csv(DATA_FOLDER / (tag + '.csv'), comment='#')
    if filter_y is not None:
        df = df[df[plot_column].isin(filter_y)]
    if drop_names is not None:
        df = df[~df['Name'].isin(drop_names)]
    if include_only_names is not None:
        df = df[df['Name'].isin(include_only_names)]

    default_labels = DEFAULT_LABEL_MAP.copy()
    default_labels.update(labels or {})
    labels = default_labels
    if bwd_df is None:
        bwd_path = DATA_FOLDER / (tag + '-bwd.csv')
        if bwd_path.exists():
            bwd_df = pd.read_csv(bwd_path, comment='#')
        else:
            print(f"Could not find backward file {bwd_path}.")
    if bwd_df is not None:
        if filter_y is not None:
            bwd_df = bwd_df[bwd_df[plot_column].isin(filter_y)]
        if drop_names is not None:
            bwd_df = bwd_df[~bwd_df['Name'].isin(drop_names)]
        if include_only_names is not None:
            bwd_df = bwd_df[bwd_df['Name'].isin(include_only_names)]

        make_plot(df, f"{plot_title}:  BWD + FWD", label_map=labels, plot_column=plot_column,
                  bwd_df=bwd_df, xlabel=xlabel, col_order=col_order,
                  legend_outside=legend_outside, skip_timestamp=skip_timestamp, color_map=color_map)
    else:
        make_plot(df, f"{plot_title}: forward pass", label_map=labels, plot_column=plot_column,
                  skip_timestamp=skip_timestamp, xlabel=xlabel, color_map=color_map,
                  col_order=col_order)


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


def plot_block_sizes(df=None, bwd_df=None, tag=None, name=None):
    if df is None:
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

    if bwd_df is None:
        dfs = []
        for sz in ['', '-64-8-1', '-512-1-1', '-128-8-1']:
            df_temp = pd.read_csv(tag + sz + '-bwd.csv')
            dace_rows = df_temp['Name'] == 'dace_autoopt_persistent_mem_csr'
            df_temp['Name'][dace_rows] = df_temp['Name'][dace_rows] + sz
            dfs.append(df_temp)
        bwd_df = pd.concat(dfs)

    block_sizes = ['1024_1_1', '32_1_1', '64_8_1', '512_1_1']
    formats = ['csr', 'coo']
    labels = {
        f'dace_{sz}_{fmt}_adapt': f'DaCe {sz.replace("_", ",")} {fmt} adapt' for
        sz in block_sizes for fmt in formats}
    labels.update(block_size_labels)
    make_plot(df, f"{name}: Block sizes (fwd + bwd)",
              label_map=labels, bwd_df=bwd_df)


if __name__ == '__main__':
    main()
