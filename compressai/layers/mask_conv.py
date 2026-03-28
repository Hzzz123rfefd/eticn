import os
import sys
sys.path.append(os.getcwd())
from typing import Any

import torch
import torch.nn as nn

from torch import Tensor

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)
    
class MaskedConv1d(nn.Conv1d):
    r"""Masked 1D convolution implementation, mask future "unseen" positions.
    Useful for building auto-regressive sequence models.

    mask_type:
        'A' -> mask current position
        'B' -> keep current position
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        # 注册 mask
        self.register_buffer("mask", torch.ones_like(self.weight.data))

        _, _, k = self.mask.size()   # kernel_size
        center = k // 2

        # 核心mask逻辑
        # A: 当前点也mask掉
        # B: 当前点保留
        self.mask[:, :, center + (mask_type == "B") :] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x)