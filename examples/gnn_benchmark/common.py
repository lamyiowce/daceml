import torch


def get_loss_and_targets(model_name, val_dtype, data, num_classes):
    if 'single_layer' in model_name:
        loss_fn = lambda pred, targets: torch.sum(pred)
        targets = torch.clone(data.y, memory_format=torch.contiguous_format)
        targets.requires_grad = False
    else:
        loss_fn = torch.nn.MSELoss()
        targets = torch.clone(
            torch.nn.functional.one_hot(data.y, num_classes=num_classes).to(val_dtype),
            memory_format=torch.contiguous_format)
        targets.requires_grad = False
    return loss_fn, targets
