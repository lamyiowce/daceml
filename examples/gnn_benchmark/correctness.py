import copy
from typing import Dict, Sequence, Tuple

import torch

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
        if parameter.grad is None:
            print(f"No gradient computed for {name} in {name_result}.")
            all_correct = False
        if result_grad is None:
            print(f"No gradient computed for {name} in {name_expected}.")
            all_correct = False
        all_correct &= check_equal(result_grad, parameter.grad,
                                   name_expected=name_expected + ": " + name,
                                   name_result=name_result + ": " + name,
                                   verbose=verbose)
    return all_correct


def check_correctness(dace_models: Dict[str, 'ExperimentInfo'],
                      torch_experiments: Sequence[
                          Tuple[str, torch.nn.Module, Sequence[torch.Tensor]]],
                      targets: torch.Tensor,
                      backward: bool,
                      loss_fn: torch.nn.Module) -> bool:
    model_set = set()

    def copy_if_duplicate(
            entry: Tuple[str, torch.nn.Module, Sequence[torch.Tensor]]):
        name, model, inputs = entry
        if model in model_set:
            model = copy.deepcopy(model)
        model_set.add(model)
        return name, model, inputs

    torch_experiments = [copy_if_duplicate(entry) for entry in
                         torch_experiments]

    torch_preds = []
    for name, model, inputs in torch_experiments:
        model.train()
        torch_preds.append(model(*inputs))

    if torch_preds:
        reference_pred = torch_preds[0]
        reference_name, reference_model, reference_input = torch_experiments[0]
        reference_input_features = reference_input[0]
        for (name, model, _,), pred in zip(torch_experiments[1:],
                                           torch_preds[1:]):
            check_equal(pred, reference_pred, name_result=name,
                        name_expected=reference_name)
    else:
        reference_pred = None

    def backward_func(pred):
        loss = loss_fn(pred, targets)
        loss.backward()

    if backward:
        for pred in torch_preds:
            backward_func(pred)

        if USE_GPU:
            torch.cuda.synchronize()

        for (name, model, inputs,), pred in zip(torch_experiments[1:],
                                           torch_preds[1:]):
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            check_gradients(model, reference_model, name, reference_name,
                            verbose=True)
            if reference_input_features.requires_grad:
                check_equal(result=inputs[0].grad,
                            expected=reference_input_features.grad,
                            name_result=name + ": input_features.grad",
                            name_expected=reference_name + ": input_features.grad",
                            verbose=True)

    if dace_models:
        from examples.gnn_benchmark.util import register_replacement_overrides

        for name, experiment_info in dace_models.items():
            print(f"---> Checking correctness for {name}...")
            model = experiment_info.model_eval
            args = experiment_info.data.to_input_list()
            for i, array in enumerate(args):
                assert array.is_contiguous(), f"{i}th input not contiguous!"
            register_replacement_overrides(experiment_info.impl_name,
                                           experiment_info.gnn_type,
                                           experiment_info.idx_dtype,
                                           experiment_info.val_dtype)

            if USE_GPU:
                torch.cuda.nvtx.range_push(name + ' forward correctness')
                torch.cuda.synchronize()
            dace_pred = model(*args)
            if USE_GPU:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()

            if reference_pred is not None:
                experiment_info.correct = check_equal(dace_pred,
                                                      reference_pred,
                                                      name_result=f"Forward predictions for DaCe {name}",
                                                      name_expected=reference_name)
            else:
                print(
                    f"Not checking correctness for {name} because no torch prediction was computed.")

            if backward:
                model = experiment_info.model_train
                if USE_GPU:
                    torch.cuda.nvtx.range_push(
                        name + ' backward correctness (pred)')
                    torch.cuda.synchronize()
                dace_pred = model(*args)
                if USE_GPU:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(
                        name + ' backward correctness (grad)')
                    torch.cuda.synchronize()
                backward_func(dace_pred)
                if USE_GPU:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()

                if reference_pred is not None:
                    check_equal(dace_pred,
                                reference_pred,
                                name_result=f"Backward predictions for DaCe {name}",
                                name_expected=reference_name)
                    experiment_info.correct_grads = check_gradients(model.model,
                                                                    reference_model,
                                                                    name_result=f"Gradients for DaCe {name}",
                                                                    name_expected=reference_name + "gradients")
                    if reference_input_features.requires_grad:
                        check_equal(result=args[0].grad,
                                    expected=reference_input_features.grad,
                                    name_result=f"input_features.grad for DaCe {name}",
                                    name_expected=reference_name + ": input_features.grad",
                                    verbose=True)

    correct_keys = [key for key, value in dace_models.items() if value.correct]
    incorrect_keys = [key for key, value in dace_models.items() if
                      value.correct == False]

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
                                value.correct_grads == False]
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
