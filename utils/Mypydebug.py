"""
This module is used to show the shape of params when debugging.
"""

import torch

original_repr = torch.Tensor.__repr__
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}'

def show_shape():
    torch.Tensor.__repr__ = custom_repr





