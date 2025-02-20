import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import yaml
from torch import Tensor

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def positionalencoding1d(batch,max_len,d_model):
    encoding = torch.zeros(max_len, d_model).float()
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, max_len)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)
    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    if d_model % 2 == 0:
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    else:
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i[0:-1] / d_model)))

    return encoding.unsqueeze(0).repeat(batch,1,1)

def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    


def calculate_psnr(img1, img2):

    assert img1.shape == img2.shape, "输入图像的形状必须相同"

    img1 = img1.float()
    img2 = img2.float()

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  

    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr.item()

def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None

def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)