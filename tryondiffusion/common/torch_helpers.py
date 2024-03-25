from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from tryondiffusion.common.python_helpers import exists, split_iterable


def module_device(module: nn.Module) -> torch.device:
    """Get the device of a PyTorch module."""
    return next(module.parameters()).device


def zero_init_(m: nn.Module) -> nn.Module:
    """Initialize weights and biases of a module to zeros."""
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)
    return m


def eval_decorator(fn: Callable) -> Callable:
    """Decorator to set a PyTorch model to evaluation mode during function execution."""
    def inner(model: nn.Module, *args, **kwargs) -> Any:
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def cast_torch_tensor(fn: Callable, cast_fp16: bool = False) -> Callable:
    """Decorator to cast input tensors to PyTorch tensors and move them to the correct device."""
    @wraps(fn)
    def inner(model: nn.Module, *args, **kwargs) -> Any:
        device = kwargs.pop("_device", module_device(model))
        cast_device = kwargs.pop("_cast_device", True)

        should_cast_fp16 = cast_fp16 and model.cast_half_at_training

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(
            map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args)
        )

        if cast_device:
            all_args = tuple(
                map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args)
            )

        if should_cast_fp16:
            all_args = tuple(
                map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t,
                    all_args)
            )

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(zip(kwargs_keys, kwargs_values))

        out = fn(model, *args, **kwargs)
        return out
    return inner


def split(t: torch.Tensor, split_size: int = None) -> Any:
    """
    Split a tensor along the first dimension or iterables into chunks of a specified size.

    Args:
        t (torch.Tensor or Iterable): The tensor or iterable to split.
        split_size (int, optional): The size of each chunk. If None, no split is performed.

    Returns:
        Any: The result of the split operation, which could be a tensor or a list of chunks.
    """
    if split_size is None:
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim=0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    raise TypeError("Unsupported type for splitting")


