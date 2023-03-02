import torch

from ...encoding_objects.dyn_cart_mri_enc_obj import Dyn2DCartEncObj
from ...networks.q_cart_mri_primal_dual_nn import QCartPrimalDualNN
from ...data.qmri.h5 import hdf5DS
from ...data.qmri import QCartUSDataset, T1Inversion, T1Saturation
from ...utils.save_animation import save_animation
from ...utils.maskedloss import maskedloss

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.optimize
from typing import Tuple, Union
from functools import partial


def search1d(func, low_bound: float = 1e-4, up_bound: float = 1, steps0: int = 10, steps1: int = 3):
    """
    Performs a log gridsearch between low_bound and up_bound with steps0 in between
    followed by steps1 steps scalar optimisation of Brents methods
    """

    def find_interval(y, x=None):
        if x is None:
            x = np.arange(len(y))
        lowest = np.argmin(y)
        if lowest == 0:
            return (x[0] / 2, x[0])
        elif lowest == len(x) - 1:
            return (x[-1], x[-1] * 2)
        elif y[lowest - 1] > y[lowest + 1]:

            return (x[lowest], x[lowest + 1])
        else:
            return (x[lowest - 1], x[lowest])

    lambdas = 10 ** np.linspace(np.log10(max(low_bound, 10 ** (-steps0))), np.log10(up_bound), steps0)
    r = [func(l) for l in lambdas]
    (
        low_bound,
        up_bound,
    ) = find_interval(r, lambdas)
    res = scipy.optimize.minimize_scalar(func, bounds=[low_bound, up_bound], method="bounded", options={"maxiter": steps1})
    return res.x, res.fun, np.array(r), lambdas


def search2d(
    func,
    low_bound: Union[float, Tuple[float, float]] = (1e-5, 1e-4),
    up_bound: Union[float, Tuple[float, float]] = (0.2, 0.05),
    steps0: Union[int, Tuple[int, int]] = (10, 5),
    steps1: int = 3,
):
    """
    Performs a 2D log gridsearch between low_bound[0],lowbound[1] and up_bound[0],up_bound[1] with steps0 in between
    followed by steps1 steps of Nelder-Mead
    """

    def totuple(x):
        if np.isscalar(x):
            return (x, x)
        if len(x) == 1:
            return (x[0], x[0])
        return tuple(x)

    low_bound, up_bound, steps0 = (totuple(i) for i in (low_bound, up_bound, steps0))

    lambdas0 = 10 ** np.linspace(np.log10(max(low_bound[0], 10 ** (-steps0[0]))), np.log10(up_bound[0]), steps0[0])
    lambdas1 = 10 ** np.linspace(np.log10(max(low_bound[1], 10 ** (-steps0[1]))), np.log10(up_bound[1]), steps0[1])
    lambdas = np.stack(np.meshgrid(lambdas0, lambdas1))

    r = np.array([func(l) for l in lambdas.reshape(2, -1).T]).reshape(lambdas.shape[1:])

    i0, i1 = np.unravel_index(np.argmin(r), r.shape)
    best = lambdas[:, i0, i1]

    if i0 == 0:
        l1 = best[1] / 2
        u1 = lambdas[1, i0 + 1, i1]
    elif i0 == lambdas.shape[1] - 1:
        l1 = lambdas[1, i0 - 1, i1]
        u1 = best[1] * 2
    else:
        l1 = lambdas[1, i0 - 1, i1]
        u1 = lambdas[1, i0 + 1, i1]

    if i1 == 0:
        l0 = best[0] / 2
        u0 = lambdas[0, i0, i1 + 1]
    elif i1 == lambdas.shape[2] - 1:
        l0 = lambdas[0, i0, i1 - 1]
        u0 = best[0] * 2
    else:
        l0 = lambdas[0, i0, i1 - 1]
        u0 = lambdas[0, i0, i1 + 1]

    bounds = scipy.optimize.Bounds((l0, l1), (u0, u1), (True, True))
    res = scipy.optimize.minimize(func, best, method="Nelder-Mead", bounds=bounds, tol=None, callback=None, options={"maxfev": 20})
    return res.x, res.fun, r, lambdas


def mode(modestr):
    validmodes = (
        "lambda_xyt",
        "lambda_xy_t",
    )
    if modestr not in validmodes:
        raise ValueError
    return modestr


parser = argparse.ArgumentParser()
parser.add_argument("--nu", default=[64], nargs="+", type=int)
parser.add_argument("--datapath", default="data/", type=str)
parser.add_argument("--imagesize", default=192, type=int)
parser.add_argument("--acc_factor", default=[4, 8], type=int, nargs="+")
parser.add_argument("--Ncoils", default=8, type=int)
parser.add_argument("--mode", default="lambda_xy_t", type=mode)
parser.add_argument("--low_bound", default=[5e-4, 5e-4], type=float, nargs="+")
parser.add_argument("--up_bound", default=[0.05, 0.05], type=float, nargs="+")
parser.add_argument("--steps0", default=[20, 20], type=int, nargs="+")
parser.add_argument("--steps1", default=20, type=int)
parser.add_argument("--maskweight", default=0.0, type=float)
parser.add_argument("--neptune", default=False, action="store_true")
parser.add_argument("--savefile", default="./model_grid.pt", type=str)
parser.add_argument("--signalfunction", default="T1Inversion", type=str)
args = parser.parse_args()


if args.neptune:
    import neptune.new as neptune

    run = neptune.init()
    # API Token and project set as env variable
    run["parameters"] = vars(args)

signalfunction = globals()[args.signalfunction]
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    ds = hdf5DS(args.datapath)
except:
    ds = QCartUSDataset(signalfunction(), args.datapath, acceleration=args.acc_factor, Ncoils=args.Ncoils, Npx=args.imagesize, random_phase=True)
if not len(ds):
    raise ValueError(f"Unable to open training data at {args.datapath}")
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True)

EncObj = Dyn2DCartEncObj()

netsettings = dict()
pdhg = QCartPrimalDualNN(EncObj, CNN_block=None, nu=args.nu[-1], mode=args.mode).to(device=device)

pbar = tqdm(dl)


def test(lambdas, fargs, mask, y):
    with torch.no_grad():
        old = pdhg.set_lambda(torch.as_tensor(lambdas))
        xTV, *_ = pdhg(*fargs)
        loss = maskedloss(xTV, y, mask.unsqueeze(-1), maskweight=args.maskweight)
        pdhg.lambda_raw.data = old
        ret = loss.sum().item()
    return ret


opt_lambdas = []
opt_losses = []
test_lambdas = None
test_losses = []
for batchid, batch in enumerate(pbar):
    x, y, mask, classes, noise, csm, kmask = (i.to(device=device) for i in batch)
    ku = EncObj.apply_A(y, csm, kmask) + EncObj.apply_mask(noise, kmask)
    xu = EncObj.apply_AH(ku, csm, kmask)
    func = partial(test, fargs=(xu, ku, kmask, csm), mask=mask, y=y)

    if args.mode == "lambda_xyt":
        lam, loss, lamdas_losses, test_lambdas = search1d(func, args.low_bound[0], args.up_bound[0], args.steps0[0], args.steps1)
        spatial, temporal = lam, lam
    else:
        lam, loss, lamdas_losses, test_lambdas = search2d(func, args.low_bound, args.up_bound, args.steps0, args.steps1)
        spatial, temporal = lam[0], lam[1]

    opt_lambdas.append(torch.tensor(lam, dtype=torch.float32))
    test_losses.append(torch.tensor(lamdas_losses, dtype=torch.float32))

    if args.neptune:
        run["loss"].log(loss)
        run["lambda/spatial/min"].log(spatial)
        run["lambda/spatial/max"].log(spatial)
        run["lambda/temporal/min"].log(temporal)
        run["lambda/temporal/max"].log(temporal)
        run["nu"].log(pdhg.nu)
        run["sigma"].log(pdhg.sigma)
        run["tau"].log(pdhg.tau)
        run["theta"].log(pdhg.theta)

    pbar.set_postfix({"loss": f"{loss:.4f}", "spatial": f"{spatial:.4f}", "temp": f"{temporal:.4f}"})

pdhg.cpu()
tosave = dict(state=pdhg.state_dict(), **vars(args), opt_lambdas=torch.stack(opt_lambdas), test_lambdas=torch.as_tensor(test_lambdas), test_losses=torch.stack(test_losses))
torch.save(tosave, args.savefile)
print(f"saved to {args.savefile}")

if args.neptune:
    run.stop()
