import os
import torch.autograd as ta

from training.multi_device_utils import is_device_manually_assigned

orig_backward = ta.backward


def custom_backward(
    tensors,
    grad_tensors=None,
    retain_graph=None,
    create_graph=False,
    grad_variables=None,
    inputs=None,
):
    if is_device_manually_assigned():
        retain_graph = True

    return orig_backward(
        tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs
    )


ta.backward = custom_backward
