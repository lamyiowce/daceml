from pathlib import Path

import pandas as pd

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
    'dace_csr_coo_adapt': 'Dace CSR/COO adapt, CSR 50%',
    'dace_csr_coo': 'Dace CSR/COO, CSR 50%',
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
