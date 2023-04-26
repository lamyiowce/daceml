import copy
from typing import Dict

import torch

from examples.gnn_benchmark.util import register_replacement_overrides

USE_GPU = torch.cuda.is_available()

def check_equal(result, expected, name_result=None, name_expected=None,
                verbose=True):
    name_result = name_result or 'result'
    name_expected = name_expected or 'expected'
    if torch.allclose(expected, result, atol=1.0e-5):
        if verbose:
            print(
                f"==== Correct: {name_result}.  ☆ ╰(o＾◡＾o)╯ ☆  ({abs((expected - result)).max().item()}) ====")
    else:
        print(
            f"****** INCORRECT: {name_result}! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print("** Max abs error: ",
              abs((expected - result)).max().item())
        print("** Avg abs error: ",
              abs((expected - result)).mean().item())
        print("** Max rel error: ",
              (abs((expected - result)) / abs(
                  result)).max().item())
        print("** Avg rel error: ", (abs((expected - result)) / abs(
            result)).mean().item())
        print(f"** {name_result}:", result)
        print(f"** {name_expected}:", expected)
        return False

    return True


def check_gradients(result_model: torch.nn.Module,
                    expected_model: torch.nn.Module,
                    name_result: str,
                    name_expected: str,
                    verbose=True) -> bool:
    result_parameters = dict(result_model.named_parameters())
    all_correct = True
    for name, parameter in expected_model.named_parameters():
        result_grad = result_parameters[name].grad
        all_correct &= check_equal(result_grad, parameter.grad,
                                   name_expected=name_expected + ": " + name,
                                   name_result=name_result + ": " + name,
                                   verbose=verbose)
    return all_correct


def check_correctness(dace_models: Dict[str, 'ExperimentInfo'],
                      torch_model: torch.nn.Module,
                      torch_edge_list_args,
                      torch_csr_args,
                      targets: torch.Tensor,
                      backward: bool) -> bool:
    torch_model_csr = copy.deepcopy(torch_model)
    torch_model.train()
    torch_model_csr.train()

    torch_edge_list_pred = torch_model(*torch_edge_list_args)
    torch_csr_pred = torch_model_csr(*torch_csr_args)
    check_equal(torch_csr_pred.detach(), torch_edge_list_pred.detach(),
                verbose=False)

    def backward_func(pred):
        loss = criterion(pred, targets)
        loss.backward()

    if backward:
        if hasattr(torch_model, 'conv2'):
            criterion = torch.nn.NLLLoss()
        else:
            criterion = lambda pred, targets: torch.sum(pred)

        backward_func(torch_edge_list_pred)
        backward_func(torch_csr_pred)

        check_gradients(torch_model_csr, torch_model, "CSR", "EdgeList",
                        verbose=False)

    for name, experiment_info in dace_models.items():
        print(f"---> Checking correctness for {name}...")
        model = experiment_info.model
        args = experiment_info.data.to_input_list()
        register_replacement_overrides(experiment_info.impl_name,
                                       experiment_info.gnn_type,
                                       experiment_info.idx_dtype,
                                       experiment_info.val_dtype)

        if USE_GPU:
            torch.cuda.nvtx.range_push(name + ' forward correctness')
        dace_pred = model(*args)
        if USE_GPU:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        experiment_info.correct = check_equal(dace_pred,
                                              torch_edge_list_pred,
                                              name_result=f"Predictions for DaCe {name}",
                                              name_expected="Torch predictions")
        if backward:
            if USE_GPU:
                torch.cuda.nvtx.range_push(name + ' backward correctness')
            backward_func(dace_pred)
            if USE_GPU:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            experiment_info.correct_grads = check_gradients(model.model,
                                                            torch_model,
                                                            name_result=f"Gradients for DaCe {name}",
                                                            name_expected="Torch gradients")

    correct_keys = [key for key, value in dace_models.items() if value.correct]
    incorrect_keys = [key for key, value in dace_models.items() if
                      not value.correct]

    print(f"\n☆ =================== SUMMARY ================== ☆")
    if len(correct_keys) > 0:
        print(f"==== Predictions correct for {', '.join(correct_keys)}"
              f". ☆ ╰(o＾◡＾o)╯ ☆ ====")
        print(f"☆ ============================================== ☆")
    if len(incorrect_keys) > 0:
        print(f"****** INCORRECT PREDICTIONS FOR {', '.join(incorrect_keys)}"
              f"! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
        print(f"****************************************************\n")

    if backward:
        grads_correct_keys = [key for key, value in dace_models.items() if
                              value.correct_grads]
        grads_incorrect_keys = [key for key, value in dace_models.items() if
                                not value.correct_grads]
        if len(grads_correct_keys) > 0:
            print(
                f"==== Gradients correct for {', '.join(grads_correct_keys)}"
                f". ☆ ╰(o＾◡＾o)╯ ☆ ====")
            print(f"☆ ============================================== ☆")
        if len(grads_incorrect_keys) > 0:
            print(
                f"****** INCORRECT GRADIENTS FOR {', '.join(grads_incorrect_keys)}"
                f"! (ノಥ﹏ಥ)ノ彡┻━┻ ******")
            print(f"****************************************************\n")
        return len(incorrect_keys) == 0 and len(grads_incorrect_keys) == 0

    return len(incorrect_keys) == 0
