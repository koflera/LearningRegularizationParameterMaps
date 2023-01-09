import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .grad_ops import GradOperators
from ..utils.conjugate_gradient import conj_grad

"""
QMRI Application Example 
"""

def getpadding(size, multipleof=1, minpad=0):
    if (size + 2 * minpad) % multipleof == 0:
        pad = 2 * minpad
    else:
        pad = (2 * minpad) + multipleof - (size + (2 * minpad)) % multipleof
    return (math.ceil(pad / 2), math.floor(pad / 2))


class QCartPrimalDualNN(nn.Module):
    def __init__(self, EncObj, nu=64, CNN_block=None, mode="lambda_xy_t", low_bound=0.0, up_bound=None, gradmode="forward", padding="reflect", prestep_CP=None, prestep_CG=5):

        super().__init__()

        self.EncObj = EncObj
        dim = 3
        self.GradOps = GradOperators(dim, gradmode, padding)

        op_norm_A = 1
        op_norm_G = (dim * 4) ** (1 / 2)
        self.L = (op_norm_A ** 2 + op_norm_G ** 2) ** (1 / 2)
        if mode == "lambda_xyt":  # one single lambda for x,y and t
            self.lambda_raw = nn.Parameter(torch.zeros(1), requires_grad=True)
            low_bound = None if low_bound is None else torch.as_tensor(low_bound).min()
            up_bound = None if up_bound is None else torch.as_tensor(up_bound).max()
        elif mode == "lambda_xy_t":  # one /shared) lambda for x,y and one lambda for t
            self.lambda_raw = nn.Parameter(torch.zeros(2), requires_grad=True)
        elif mode == "lambda_xy_t_cnn":
            ...
        elif mode == "lambda_cnn":
            ...
        else:
            raise ValueError(f"unkown mode {mode}")
        low_bound = None if low_bound is None else torch.as_tensor(low_bound)
        up_bound = None if up_bound is None or up_bound==0.0 else torch.as_tensor(up_bound)
        self.mode = mode
        self.register_buffer("low_bound", low_bound, False)
        self.register_buffer("up_bound", up_bound, False)
        self.cnn = CNN_block
        self.prestep_CP = prestep_CP
        self.prestep_CG = prestep_CG
        self.padding = padding
        self.padmultiple, self.minpad = 4, 4
        self.nu = nu

        # sigma, tau scaling constants:
        # sigma = sigmoid(alpha)/L * exp(beta)
        # tau = sigmoid(alpha)/L / exp(beta)
        # ensuring sigma*tau*L^2 < 1 for convergence
        #  theta=sigmoid(theta_raw) as theta should be in \in [0,1]. Starting theta close to 1.
        self.alpha = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.theta_raw = nn.Parameter(torch.tensor(5.0), requires_grad=True)

    @property
    def sigma(self):
        return torch.sigmoid(self.alpha) / self.L * torch.exp(self.beta)

    @property
    def tau(self):
        return torch.sigmoid(self.alpha) / self.L * torch.exp(-self.beta)

    @property
    def theta(self):
        return torch.sigmoid(self.theta_raw)

    def sample_prestep_CP(self):
        if self.prestep_CP is None:
            return 0
        elif isinstance(self.prestep_CP):
            return self.prestep_CP
        else:
            return torch.randint(*self.prestep_CP)

    @staticmethod
    def solve_normal_eqs(Ahy, AHA, niter=0):
        # solve a few few steps of CG for solving normal equations
        # to get better initialization
        if niter == 0:
            return Ahy.detach().clone()
        with torch.no_grad():
            return conj_grad(AHA, Ahy, Ahy, niter=niter)

    def prepare_for_cnn(self, x):
        # convert to 2-channel view: (Nb,Nx,Ny,Nt) (complex) --> (Nb,2,Ny,Ny,Nt) (real)
        x_real = torch.view_as_real(x).moveaxis(-1, 1)
        padsizes = [getpadding(n, self.padmultiple, self.minpad) for n in x_real.shape[2:]]
        pad = [s for p in padsizes[::-1] for s in p]  # pad takes the padding size starting from last dimension moving forward
        crop = [Ellipsis] + [slice(p[0], -p[1]) for p in padsizes]
        x_padded_real = F.pad(x_real, pad, mode=self.padding)
        return x_padded_real, crop

    def scale_lambda(self, lambdas):
        low_bound = 0.0 if self.low_bound is None else self.low_bound
        if self.up_bound is not None:
            return low_bound + (self.up_bound - low_bound) * torch.sigmoid(lambdas)
        else:
            return low_bound + torch.nn.functional.softplus(lambdas, beta=5)

    def unscale_lambda(self, lambdas):
        def invsoftplus(x, beta=1):
            x = x * beta
            x = x + torch.log(-torch.expm1(-x))
            x = x / beta
            return x

        def invsigmoid(x):
            return torch.log(x / (1 - x))

        low_bound = 0.0 if self.low_bound is None else self.low_bound
        if self.up_bound is not None:
            return invsigmoid((lambdas - low_bound) / (self.up_bound - low_bound))
        else:
            return invsoftplus(lambdas - low_bound, beta=5)

    def set_lambda(self, x, raw=False):
        if not self.mode in ("lambda_xyt", "lambda_xy_t"):
            raise NotImplementedError("only for lambda_xyt or lambda_xy_t possible to set lambda")
        old = self.lambda_raw.data.clone() if raw else self.scale_lambda(self.lambda_raw)
        lambdas = x.clone().to(device=self.lambda_raw.device, dtype=self.lambda_raw.dtype) if raw else self.unscale_lambda(x.to(device=self.lambda_raw.device, dtype=self.lambda_raw.dtype))
        self.lambda_raw.data[:] = lambdas
        return old

    def get_lambda_map_cnn(self, x):
        x_padded_real, crop = self.prepare_for_cnn(x)
        lambda_cnn = self.cnn(x_padded_real)
        lambda_cnn = lambda_cnn[crop]
        lambda_scaled = self.scale_lambda(lambda_cnn)
        lambda_t, lambda_xy = torch.chunk(lambda_scaled, 2, 1)
        lambda_reg = torch.cat((lambda_xy, lambda_xy, lambda_t), dim=1)
        return lambda_reg

    def get_lambda_scalar_cnn(self, x):
        x_padded_real, _ = self.prepare_for_cnn(x)  # padding and comlez->real
        lambda_cnn = self.cnn(x_padded_real)
        lambda_scaled = self.scale_lambda(lambda_cnn)
        lambda_t, lambda_xy = torch.chunk(lambda_scaled, 2, -1)
        lambda_reg = torch.cat((lambda_xy, lambda_xy, lambda_t), dim=1)[..., None, None, None]
        return lambda_reg

    @staticmethod
    def pdhg(nu, x0, y, sigma, tau, theta, lambda_reg, A, AH, G, GH, return_state=False):
        from .prox_ops import clipact

        """
            implemented according to
            https://iopscience.iop.org/article/10.1088/0031-9155/57/10/3065/pdf, algorithm 4
            """
        # workaround for mixed complex/real addcmul for pytorch <= 1.13 compatibility
        if y.is_complex():
            addcmul = lambda x1, x2, x3: torch.view_as_complex(torch.addcmul(torch.view_as_real(x1), torch.view_as_real(x2), x3))
        else:
            addcmul = torch.addcmul

        if isinstance(x0, dict):
            p, q, xbar, x1 = x0["p"], x0["q"], x0["xbar"], x0["x1"]
        else:
            p = torch.zeros(y.shape, dtype=y.dtype, device=x0.device)
            q = torch.zeros(x0.shape[0], 3, *x0.shape[1:], dtype=x0.dtype, device=x0.device)
            xbar = x0
            x1 = x0

        for ku in range(nu):
            p = addcmul(p, A(xbar) - y, sigma)
            p = p * (1 / (1 + sigma))
            q = addcmul(q, G(xbar), sigma)
            q = clipact(q, lambda_reg)
            step = AH(p) + GH(q)
            x1 = addcmul(x1, step, -tau)
            xbar = addcmul(x1, step, -tau * theta)
        if return_state:
            return x1, {"p": p, "q": q, "xbar": xbar, "x1": x1}
        return x1

    def forward(self, x, y, mask, csmap=None):

        # first approximately solve the normal equations to get a better estimate of the image
        x0 = self.solve_normal_eqs(x, AHA=lambda x: self.EncObj.apply_AHA(x, csmap, mask), niter=self.prestep_CG)

        if self.mode == "lambda_xyt":
            # one constant lambda
            lambda_xyt = self.scale_lambda(self.lambda_raw)[0]
            lambda_reg = torch.stack([lambda_xyt, lambda_xyt, lambda_xyt])[None, :, None, None, None]


        elif self.mode == "lambda_xy_t":
            # one spatial, one tempoeral constant lambda
            lambda_xy, lambda_t = self.scale_lambda(self.lambda_raw)
            lambda_reg = torch.stack([lambda_xy, lambda_xy, lambda_t])[None, :, None, None, None]

        elif self.mode == "lambda_xy_t_cnn":
            # estimate scalar lambda reg from the image
            lambda_reg = self.get_lambda_scalar_cnn(x0)

        elif self.mode == "lambda_cnn":
            # estimate lambda map from the image
            lambda_reg = self.get_lambda_map_cnn(x0)

        else:
            raise NotImplementedError(f"unknown mode {self.mode}")

        if self.nu < 1:
            # only CG
            return x0, lambda_reg

        A = lambda x: self.EncObj.apply_A(x, csmap, mask)
        AH = lambda x: self.EncObj.apply_AH(x, csmap, mask)
        G = self.GradOps.apply_G
        GH = self.GradOps.apply_GH

        if self.training and self.prestep_CP is not None:
            # do some steps of CP without taking the gradient, as initialisation.
            prestep_CP = self.sample_prestep_CP()
            with torch.no_grad():
                x1 = self.pdhg(prestep_CP, x0, y, self.sigma, self.tau, self.theta, lambda_reg, A, AH, G, GH, return_state=True)
            ret = self.pdhg(self.nu - prestep_CP, x1, y, self.sigma, self.tau, self.theta, lambda_reg, A, AH, G, GH, return_state=False)
            return ret, lambda_reg

        ret = self.pdhg(self.nu, x0, y, self.sigma, self.tau, self.theta, lambda_reg, A, AH, G, GH)
        return ret, lambda_reg
