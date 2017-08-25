"""Helpful functions."""

import torch
from torch.autograd import Variable


def make_variable(tensor):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def make_cuda(model):
    """Use CUDA if available."""
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def concat_feat(src, tgt):
    """Concatate features."""
    if src is None:
        out = tgt
    else:
        out = torch.cat([src, tgt])
    return out
