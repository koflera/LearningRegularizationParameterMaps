import torch
import torch.nn as nn
import torch.nn.functional as F

from .grad_ops import GradOperators
from .prox_ops import ClipAct


class DynamicImagePrimalDualNN(nn.Module):
    def __init__(
        self,
        T=128,
        cnn_block=None,
        mode="lambda_cnn",
        up_bound=0,
        phase="training",
    ):
        super(DynamicImagePrimalDualNN, self).__init__()

        # gradient operators and clipping function
        dim = 3
        self.GradOps = GradOperators(dim, mode="forward", padmode="circular")

        # operator norms
        self.op_norm_AHA = torch.sqrt(torch.tensor(1.0))
        self.op_norm_GHG = torch.sqrt(torch.tensor(12.0))
        # operator norm of K = [A, \nabla]
        # https://iopscience.iop.org/article/10.1088/0031-9155/57/10/3065/pdf,
        # see page 3083
        self.L = torch.sqrt(self.op_norm_AHA**2 + self.op_norm_GHG**2)

        # function for projecting
        self.ClipAct = ClipAct()

        if mode == "lambda_xyt":
            # one single lambda for x,y and t
            self.lambda_reg = nn.Parameter(torch.tensor([-1.5]), requires_grad=True)

        elif mode == "lambda_xy_t":
            # one (shared) lambda for x,y and one lambda for t
            self.lambda_reg = nn.Parameter(
                torch.tensor([-4.5, -1.5]), requires_grad=True
            )

        elif mode == "lambda_cnn":
            # the CNN-block to estimate the lambda regularization map
            # must be a CNN yielding a two-channeld output, i.e.
            # one map for lambda_cnn_xy and one map for lambda_cnn_t
            self.cnn = cnn_block
            self.up_bound = torch.tensor(up_bound)

        # number of terations
        self.T = T
        self.mode = mode

        # constants depending on the operators
        self.tau = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1/L
        self.sigma = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1/L

        # theta should be in \in [0,1]
        self.theta = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1

        # distinguish between training and test phase;
        # during training, the input is padded using "reflect" padding, because
        # patches are used by reducing the number of temporal points;
        # while testing, "reflect" padding is used in x,y- direction, while
        # circular padding is used in t-direction
        self.phase = phase

    def get_lambda_cnn(self, x):
        # padding
        # arbitrarily chosen, maybe better to choose it depending on the
        # receptive field of the CNN or so;
        # seems to be important in order not to create "holes" in the
        # lambda_maps in t-direction
        npad_xy = 4
        npad_t = 8
        pad = (npad_t, npad_t, npad_xy, npad_xy, npad_xy, npad_xy)

        if self.phase == "training":
            x = F.pad(x, pad, mode="reflect")

        elif self.phase == "testing":
            pad_refl = (0, 0, npad_xy, npad_xy, npad_xy, npad_xy)
            pad_circ = (npad_t, npad_t, 0, 0, 0, 0)

            x = F.pad(x, pad_refl, mode="reflect")
            x = F.pad(x, pad_circ, mode="circular")

        # estimate parameter map
        lambda_cnn = self.cnn(x)

        # crop
        neg_pad = tuple([-pad[k] for k in range(len(pad))])
        lambda_cnn = F.pad(lambda_cnn, neg_pad)

        # double spatial map and stack
        lambda_cnn = torch.cat((lambda_cnn[:, 0, ...].unsqueeze(1), lambda_cnn), dim=1)

        # constrain map to be striclty positive; further, bound it from below
        if self.up_bound > 0:
            # constrain map to be striclty positive; further, bound it from below
            lambda_cnn = self.up_bound * self.op_norm_AHA * torch.sigmoid(lambda_cnn)
        else:
            lambda_cnn = 0.1 * self.op_norm_AHA * F.softplus(lambda_cnn)

        return lambda_cnn

    def forward(self, x, lambda_map=None):
        # initial reconstruction
        mb, _, Nx, Ny, Nt = x.shape
        device = x.device

        # starting values
        xbar = x.clone()
        x0 = x.clone()
        xnoisy = x.clone()

        # dual variable
        p = x.clone()
        q = torch.zeros(mb, 3, Nx, Ny, Nt, dtype=x.dtype).to(device)

        # sigma, tau, theta
        sigma = (1 / self.L) * torch.sigmoid(self.sigma)  # \in (0,1/L)
        tau = (1 / self.L) * torch.sigmoid(self.tau)  # \in (0,1/L)
        theta = torch.sigmoid(self.theta)  # \in (0,1)

        # distinguish between the different cases
        if self.mode == "lambda_xyt":
            lambda_reg = F.softplus(self.lambda_reg)  # \in (0,\infty)

        elif self.mode == "lambda_xy_t":
            # get xy- and t-lambda
            lambda_reg_xy = torch.stack(2 * [self.lambda_reg[0]])
            lambda_reg_t = self.lambda_reg[1].unsqueeze(0)

            # conatentate xy -and t-lambda
            lambda_reg = (
                torch.cat([lambda_reg_xy, lambda_reg_t])
                .unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            lambda_reg = F.softplus(lambda_reg)

        elif self.mode == "lambda_cnn":
            if lambda_map is None:
                # estimate lambda reg from the image
                lambda_reg = self.get_lambda_cnn(x)
            else:
                lambda_reg = lambda_map

        for kT in range(self.T):
            # update p
            p =  (p + sigma * (xbar - xnoisy) ) / (1. + sigma)
            # update q
            q = self.ClipAct(q + sigma * self.GradOps.apply_G(xbar), lambda_reg)

            x1 = x0 - tau * p - tau * self.GradOps.apply_GH(q)

            if kT != self.T - 1:
                # update xbar
                xbar = x1 + theta * (x1 - x0)
                x0 = x1

        return x1
