import torch
import torch.nn as nn


class Dyn2DCartEncObj(nn.Module):

    """
    Implementation of operators for Cartesian MR image reconstruction
    (dynamics are in last dimension, batch in first)
    """

    def __init__(self, norm="ortho"):
        self.norm = norm
        super().__init__()

    def apply_C(self, x, csm):
        """
        Apply Coil Sensitivity Expand Operation
        input:
            x (mb,  Nx, Ny, Nt)
            csm (mb, Nc, Nx, Ny)
        """

        Cx = csm.unsqueeze(-1) * x.unsqueeze(1)

        return Cx

    def apply_E(self, x):
        """
        Apply Fourier Operation on (2,3) Dimension
        input:
            x (mb, Nc, Nx, Ny, Nt)
        """
        k = torch.fft.fftn(x, dim=(2, 3), norm=self.norm)

        return k

    def apply_mask(self, k, mask):
        """
        Mask the Fourier data
        input:
            k (mb, Nc, Nx, Ny, Nt)
            mask (mb, Nkx, Nky, Nt)
        """

        return k * mask.unsqueeze(1)

    def apply_A(self, x, csm, mask):
        """
        Apply A Operation, ie. image-to-k-space
        input:
            x (mb, Nc, Nx, Ny, Nt)
            csm  (mb, Nc, Nx, Ny)
            mask (mb, Nkx, Nky, Nt)
        output:
            k    (mb, Nc, Nx, Ny, Nt)
        """

        return self.apply_mask(self.apply_E(self.apply_C(x, csm)), mask)

    def apply_CH(self, xc, csm):
        """
        Apply Coil Sensitivity Reduce Operation
        input:
            x (mb, Nc, Nx, Ny, Nt)
            csm (mb,  Nc, Nx, Ny)
        """

        CHx = torch.sum(csm.conj().unsqueeze(-1) * xc, dim=1, keepdim=False)
        return CHx

    def apply_EH(self, k):
        """
        apply 2D inverse FFT
        input:
            x (mb, Nc, Nkx, Nky, Nt)
        """

        x = torch.fft.ifftn(k, dim=(2, 3), norm=self.norm)

        return x

    def apply_EC(self, x, csm):
        return self.apply_E(self.apply_C(x, csm))

    def apply_AH(self, k, csm, mask):
        """
        Apply AH Operation, ie. k-to-image
        input:
            k    (mb, Nc, Nx, Ny, Nt)
            csm  (mb, Nc, Nx, Ny)
            mask (mb, Nkx, Nky, Nt)
        output:
            x (mb, Nc, Nx, Ny, Nt)
        """

        return self.apply_CH(self.apply_EH(self.apply_mask(k, mask)), csm)

    def apply_AHA(self, x, csm, mask):
        """
        The composition A^H A used for iterative Reconstruction
        ie. image-to-image
        input:
            x (mb, Nc, Nx, Ny, Nt)
            csm  (mb, Nc, Nx, Ny)
            mask (mb, Nkx, Nky, Nt)
        output:
            x (mb, Nc, Nx, Ny, Nt)
        """
        return self.apply_AH(self.apply_A(x, csm, mask), csm, mask)
