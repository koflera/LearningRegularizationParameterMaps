import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor, tensor


def trunc_norm(mu: Union[float, Tensor], sigma: Union[float, Tensor], a: Union[float, Tensor], b: Union[float, Tensor], size: Tuple[int] = (1,)) -> Tensor:
    normal = torch.distributions.normal.Normal(0, 1)
    alpha = torch.as_tensor((a - mu) / sigma)
    beta = torch.as_tensor((b - mu) / sigma)
    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * torch.rand(size)
    v = torch.clip(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
    x = torch.clamp(x, a, b)
    return x


def random_poly2d(Nx: int, Ny: int, strength: Tuple[float], p: float = 1, sall: float = 0) -> np.ndarray:
    """
    Random 2d Polynom
    """
    if torch.rand() > p:
        return np.zeros((Nx, Ny))
    order = np.size(strength)
    c = (torch.rand((2, order))-1/2).numpy()
    x = np.linspace(-1, 1, Nx)[:, None]
    y = np.linspace(-1, 1, Ny)[None, :]
    ret = (sum(c[0, i] * x ** (i + 1) for i in range(order)) + 1) * (sum(c[1, i] * y ** (i + 1) for i in range(order)) + 1) - 1
    if sall > 0:
        ret += sall * float(trunc_norm(0, sall, -2 * sall, 2 * sall))
    return ret


def _uniform(low,high,size=1):
    return torch.rand(size)*(high-low)+low

def random_gaussians(Nx: int, Ny: int, s: float, scales: Tuple[float], p: float = 1, sall: float = 0.0) -> np.ndarray:
    """
    Random Gaussians
    """
    allg = np.zeros((Nx, Ny))
    if torch.rand() > p:
        return allg + 1
    for scalehigh, scalelow in zip(scales[1:], scales[:-1]):
        strength = _uniform(-1,1)
        sx, sy = _uniform(scalelow, scalehigh, 2)
        mx, my = _uniform(-1 + scalelow, 1 - scalelow, 2)
        alpha = _uniform(0,np.pi / 2)
        x = np.linspace(-1, 1, Nx).reshape(-1, 1) - mx
        y = np.linspace(-1, 1, Ny).reshape(1, -1) - my
        g = np.exp(-((np.cos(alpha) * x - np.sin(alpha) * y) ** 2 / sx + (np.sin(alpha) * x + np.cos(alpha) * y) ** 2 / sy))

        allg += g * strength * (sx * sy)
    rand = -np.inf
    while np.abs(rand) > 2:
        rand = float(torch.randn(1))
    allg *= rand * s / np.std(allg)
    allg -= np.mean(allg) - 1
    allg = np.clip(allg, 1 - 3 * s, 1 + 3 * s)
    if sall > 0:
        allg *= float(trunc_norm(1, sall, 1 - 2 * sall, 1 + 2 * sall, 1))
    allg = np.clip(allg, max(1e-2, 1 - 3 * s - 2 * sall), min(100, 1 + 3 * s + 2 * sall))
    return allg


def gamma_normal_noise(image: Tensor, mask: Tensor, meanvar: float = 1e-2, varvar: float = 2e-5, same_axes: Tuple[Tuple[int]] = ((-1, -2), (-3))):
    m = torch.as_tensor(meanvar, dtype=torch.float64)
    v = torch.as_tensor(varvar, dtype=torch.float64) * len(same_axes)
    d = torch.distributions.gamma.Gamma(m ** 2 / v, m / v)
    var = []
    for sa in same_axes:
        s = tensor(image.shape)
        s[(sa,)] = 1
        var.append(d.sample(s))
    var = sum(var) / len(var)
    sigma = (mask > 0) * torch.sqrt(var)
    noise = (torch.randn(image.shape) * sigma).float()
    return noise + torch.nan_to_num(image, 0, 0, 0), sigma


def cutnan(array: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    remove full-nan rows and columns (last two dimensions) of array
    """
    ind0 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 1)))
    ind1 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 2)) + (array.ndim - 1,))
    return array[..., ind1, :][..., ind0]


def AddFillMask(x):
    mask = torch.all(torch.isfinite(x), 0, True)
    return torch.cat((torch.nan_to_num(x, 0, 0, 0), mask), 0)


def RemoveFillMask(x):
    mask = x[-1:, ...] < 0.95
    x = x[:-1, ...]
    x[mask.expand(x.shape)] = np.nan
    return x
