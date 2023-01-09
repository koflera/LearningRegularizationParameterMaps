import torch
import torch.nn as nn


class ClipAct(nn.Module):
    def forward(self, x, threshold):
        return clipact(x, threshold)


def clipact(x, threshold):
    is_complex = x.is_complex()
    if is_complex:
        x = torch.view_as_real(x)
        threshold = threshold.unsqueeze(-1)
    x = torch.clamp(x, -threshold, threshold)
    if is_complex:
        x = torch.view_as_complex(x)
    return x


class ClipActOld(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, threshold):
        dtype = x.dtype
        if dtype in [torch.complex32, torch.complex64, torch.complex128]:
            x = torch.view_as_real(x)
            threshold = threshold.unsqueeze(-1)
        x = (threshold * x) / torch.max(threshold * torch.ones(x.shape).to(x.device), torch.abs(x))
        if dtype in [torch.complex32, torch.complex64, torch.complex128]:
            x = torch.view_as_complex(x)
        return x
