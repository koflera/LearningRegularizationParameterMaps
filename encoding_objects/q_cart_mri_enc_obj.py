import torch
import torch.nn as nn


class Q2DCartEncObj(nn.Module):

    """
    Implementation of operators for MR image reconstruction
    using pytorch v.1.10.1
    """

    def __init__(self, norm="ortho"):

        self.norm = norm
        super().__init__()

    def apply_C(self, x, csm):

        """
        input: 	x (mb,  Nx, Ny, Nt)
                        csm (mb, Nc, Nx, Ny)
        """

        Cx = csm.unsqueeze(-1) * x.unsqueeze(1)

        return Cx

    def apply_E(self, x):

        # apply 2D FFT
        k = torch.fft.fftn(x, dim=(2, 3), norm=self.norm)

        return k

    def apply_mask(self, k, mask):

        """
        mask the fourier data
        """

        return k * mask.unsqueeze(1)

    def apply_A(self, x, csm, mask):

        return self.apply_mask(self.apply_E(self.apply_C(x, csm)), mask)

    def apply_CH(self, xc, csm):

        """
        input: 	 x (mb, Nc, Nx, Ny, Nt)
                         csm (mb,  Nc, Nx, Ny)
        """

        CHx = torch.sum(csm.conj().unsqueeze(-1) * xc, dim=1, keepdim=False)
        return CHx

    def apply_EH(self, k):

        # apply 2D FFT
        x = torch.fft.ifftn(k, dim=(2, 3), norm=self.norm)

        return x

    def apply_EC(self, x, csm):

        return self.apply_E(self.apply_C(x, csm))

    def apply_AH(self, k, csm, mask):

        return self.apply_CH(self.apply_EH(self.apply_mask(k, mask)), csm)

    def apply_AHA(self, x, csm, mask):

        """
        the composition A^H A for the CG module
        """

        return self.apply_AH(self.apply_A(x, csm, mask), csm, mask)
