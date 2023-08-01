from pathlib import Path
from typing import Dict

import pandas as pd
import tabulate

from examples.gnn_benchmark.experiment_info import ExperimentInfo
from examples.gnn_benchmark.sdfg_util import get_total_memory, pretty_print_bytes


def check_memory(dace_experiments: Dict[str, ExperimentInfo], outfile: Path = None):
    headers = ['Model', 'Forward', 'Forward with grads', 'Backward']
    rows = []
    for name, exp_info in dace_experiments.items():
        model_fwd = exp_info.model_eval
        model_bwd = exp_info.model_train
        fwd_mem, _ = get_total_memory(model_fwd.sdfg) if model_fwd is not None else (0, 0)
        fwd_with_grad_mem, _ = get_total_memory(
            model_bwd.forward_sdfg) if model_bwd is not None else (0, 0)
        bwd_mem, _ = get_total_memory(
            model_bwd.backward_sdfg) if model_bwd is not None else (0, 0)
        rows.append([name, fwd_mem, fwd_with_grad_mem, bwd_mem])

    print("MEMORY USAGE")
    print(tabulate.tabulate(
        [[name] + [pretty_print_bytes(b) for b in vals] for name, *vals in rows],
        headers=headers,
        tablefmt='github'))

    if outfile is not None:
        path = outfile.with_name(outfile.stem + "-memory.csv")
        pd.DataFrame(rows, columns=headers).to_csv(path)
