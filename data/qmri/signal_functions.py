import torch
from typing import Optional, Tuple
from torch import nn, Tensor
import abc


class SignalFunction(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.zero = torch.zeros(1)

    @property
    @abc.abstractmethod
    def nIn(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def nOut(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        return x

    @property
    def bounds(self) -> Tensor:
        if hasattr(self, "_bounds"):
            return self._bounds[None, :, None, None, :]
        else:
            return self.zero[None, None, None, None].expand(1, self.nOut, 1, 1).clone()

    @property
    def initial_values(self) -> Tensor:
        if hasattr(self, "_initial_values"):
            return self._initial_values[None, :, None, None]
        elif hasattr(self, "_bounds"):
            self._bounds[None, :, None, None, :].mean(-1)
        else:
            return


class T1Inversion(SignalFunction):
    def __init__(self, ti: Tuple = (0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
        ti = torch.tensor(ti)[None, None, None,:]
        super().__init__()
        self.register_buffer("ti", ti)
        self.register_buffer("_bounds", torch.tensor(((-0.2, 1.2), (-0.5, 5))))
        self.register_buffer("_initial_values", torch.tensor((0.5, 0.5)))

    @property
    def nOut(self):
        return int(self.ti.numel())

    @property
    def nIn(self):
        return 2

    def forward(self, x: Tensor):
        pd, r1 = x[:, 0,...,None], x[:, 1,...,None]
        signal = pd * (1 - 2 * torch.exp(-self.ti * r1))
        return signal


class T1Saturation(SignalFunction):
    def __init__(self, ti: Tuple = (0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)):
        ti = torch.tensor(ti)[None, None, None, :]
        super().__init__()
        self.register_buffer("ti", ti)
        self.register_buffer("_bounds", torch.tensor(((-1., 2), (-1, 6))))
        self.register_buffer("_initial_values", torch.tensor((0.5, 0.5)))

    @property
    def nOut(self):
        return int(self.ti.numel())

    @property
    def nIn(self):
        return 2

    def forward(self, x: Tensor):
        pd, r1 = x[:, 0,...,None], x[:, 1,...,None]
        signal = pd * (1 - torch.exp(-self.ti * r1))
        return signal

