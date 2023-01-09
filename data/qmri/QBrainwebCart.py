from pathlib import Path
import torch
from functools import partial
import numpy as np
from torchvision import transforms as T


import sys

sys.path.append("../../../")
from utils.mrisensesim import mrisensesim
from utils.cartesian_mask_funcs import cartesian_mask

from .datasets import BrainwebSlices, values_t1t2pd3T


def random_poly2d(Nx, Ny, strength):
    order = np.size(strength)
    c = 0.5 - torch.rand(2, order) * torch.as_tensor(strength)
    x = np.linspace(-1, 1, Nx)[:, None]
    y = np.linspace(-1, 1, Ny)[None, :]
    ret = (sum(c[0, i] * x ** (i + 1) for i in range(order)) + 1) * (sum(c[1, i] * y ** (i + 1) for i in range(order)) + 1) - 1
    return ret


class CartUSDataset(torch.utils.data.Dataset):
    def __init__(self, dsx, csm_fun, kmask_fun, Ncoils=8):
        self.dsx = dsx
        self.csm_fun = csm_fun
        self.kmask_fun = kmask_fun
        self.Ncoils = Ncoils

    def __getitem__(self, idx):
        val = self.dsx[idx]
        mask, classes, x = val[0:1].bool(), val[1:2].int(), val[2:].float()
        csm = self.csm_fun(x.shape[-2:], Ncoils=self.Ncoils)
        kmask = self.kmask_fun(x.shape[-2:])

        return (
            x,
            mask,
            classes,
            csm.to(dtype=torch.complex64),
            kmask.to(dtype=torch.float32),
        )

    def __len__(self):
        return len(self.dsx)


class QCartUSDataset(torch.utils.data.Dataset):
    """
       q, path="data/brainwebClasses/",  acceleration=8, Ncoils=8, noise_strength=(0.01, 0.1), cuts=(120, 120, 0, 0, 0, 0), Npx=256, values=values_t1t2pd3T, random_phase=True
    """

    def __init__(
        self, q, path="data/brainwebClasses/", acceleration=8, Ncoils=8, noise_strength=(0.005, 0.05), cuts=(120, 120, 0, 0, 0, 0), Npx=256, values=values_t1t2pd3T, augment=True, random_phase=True
    ):

        super().__init__()
        path = Path(path)
        self._q = q
        self.noise_strength = noise_strength
        self._Ncoils = Ncoils
        self.random_phase = random_phase
        if augment:
            transforms = (
                (
                    T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(Npx / 400, Npx / 300), shear=5, fill=0.0, interpolation=T.InterpolationMode.BILINEAR),
                    T.CenterCrop((Npx, Npx)),
                    T.RandomHorizontalFlip(),
                    T.RandomHorizontalFlip(),
                ),
                (T.GaussianBlur(5, sigma=(0.01, Npx / 800)),),
            )
        else:
            transforms = (
                (T.Lambda(lambda img: T.functional.affine(img, angle=0, scale=Npx / 370, translate=(0, 0), shear=0, fill=0.0, interpolation=T.InterpolationMode.BILINEAR)), T.CenterCrop((Npx, Npx)),),
                (T.GaussianBlur(5, sigma=0.05),),
            )

        def generate_csm(imagesize, Ncoils):
            csm = torch.as_tensor(np.array(mrisensesim(imagesize, ncoils=Ncoils, coil_width=float(3 + 1 * torch.randn(1).clamp(-2, 2)), phi=float(torch.rand(1) * 360)))).abs()
            phase = torch.exp(torch.stack([2j * np.pi * (0.2 * random_poly2d(*imagesize, (1.0, 0.5, 0.25)) + torch.rand(1)) for i in range(Ncoils)]))
            csm = csm * phase
            norm = 1 / ((csm * csm.conj()).sum(0, True).sqrt())
            return csm * norm

        def qmri_cartesian_mask(imagesize, measurements, acceleration):
            if not np.isscalar(acceleration):
                if len(acceleration) == 1:
                    acceleration = float(acceleration[0])
                else:
                    acceleration = float(torch.rand(1) * (acceleration[1] - acceleration[0]) + acceleration[0])
            mask = torch.as_tensor(np.array([cartesian_mask((imagesize[0], 1), acceleration) for m in range(measurements)]), dtype=torch.float32).moveaxis(0, -1)
            mask = torch.fft.ifftshift(mask, -3)
            mask = torch.broadcast_to(mask, (*imagesize, measurements))
            return mask

        self.ds = CartUSDataset(
            BrainwebSlices(path, cuts=cuts, what=("mask", "classes", "pd", "r1",), maskval=(0, 0), transforms=transforms, classes=values),
            generate_csm,
            partial(qmri_cartesian_mask, acceleration=acceleration, measurements=self._q.nOut),
            Ncoils=Ncoils,
        )

    @staticmethod
    def phasemap(Nx, Ny, Nt):
        spatial_strength = 0.1 * (1 - 2 * torch.rand(1))
        temporal_strength = 0.05 * torch.rand(1)
        spatiotemporal_strength = 0.05 * (1 - 2 * torch.rand(1))
        global_strength = 0.2
        spatialphase = spatial_strength * random_poly2d(Nx, Ny, (1.0, 0.5, 0.25))[..., None]
        temporalphase = torch.cumsum(temporal_strength * torch.randn(Nt), 0)
        phase = torch.stack([spatiotemporal_strength * random_poly2d(Nx, Ny, (1.0, 0.5)) for i in range(Nt)], -1)
        phase = phase + spatialphase + temporalphase
        phase = (phase + (global_strength * torch.randn(1) - phase.mean())) * 2 * np.pi
        return phase

    def __getitem__(self, idx):
        x, mask, classes, csm, kmask = self.ds[idx]
        y = self._q(x[None, ...])[0]
        if self.random_phase:
            y = y * torch.exp(1j * self.phasemap(*y.shape))
        y = y.to(dtype=torch.complex64)
        if np.isscalar(self.noise_strength):
            noise_strength = self.noise_strength
        else:
            noise_strength = torch.rand(1) * (self.noise_strength[1] - self.noise_strength[0]) + self.noise_strength[0]
        noise = noise_strength * (torch.randn(self._Ncoils, *x.shape[-2:], self._q.nOut) + 1j * torch.randn(self._Ncoils, *x.shape[-2:], self._q.nOut))

        return x, y, mask.squeeze(0), classes.squeeze(0), noise, csm, kmask

    def __len__(self):
        return len(self.ds)
