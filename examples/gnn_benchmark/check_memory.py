from pathlib import Path
from typing import Dict

import pandas as pd
import tabulate

from examples.gnn_benchmark.experiment_info import ExperimentInfo
from examples.gnn_benchmark.sdfg_util import get_total_memory, pretty_print_bytes


def check_memory(dace_experiments: Dict[str, ExperimentInfo], model_name, hidden_size, outfile: Path = None):
    headers = ['Name', 'Model', 'Size', 'Forward', 'Forward with grads', 'Backward']
    rows = []
    for name, exp_info in dace_experiments.items():
        model_fwd = exp_info.model_eval
        model_bwd = exp_info.model_train
        fwd_mem, _ = get_total_memory(model_fwd.sdfg) if model_fwd is not None else (0, 0)
        fwd_with_grad_mem, _ = get_total_memory(
            model_bwd.forward_sdfg) if model_bwd is not None else (0, 0)
        bwd_mem, _ = get_total_memory(
            model_bwd.backward_sdfg) if model_bwd is not None else (0, 0)
        rows.append([name, model_name, hidden_size, fwd_mem, fwd_with_grad_mem, bwd_mem])

    print("MEMORY USAGE")
    print(tabulate.tabulate(
        [[vals[0]] + [pretty_print_bytes(b) for b in vals[-3:]] for vals in rows],
        headers=[headers[0]] + headers[-3:],
        tablefmt='github'))

    if outfile is not None:
        path = outfile.with_name(outfile.stem + "-memory.csv")
        add_header = not path.exists()
        pd.DataFrame(rows, columns=headers).to_csv(path, mode='a', header=add_header, index=False)
