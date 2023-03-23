import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from  scipy.ndimage.morphology import binary_fill_holes
from ...data.qmri import QCartUSDataset, T1Inversion, T1Saturation
from ...data.qmri.h5 import h5append, hdf5DS, to_hdf5
from ...data.qmri.QBrainwebCart import values_t1t2pd3T
from ...encoding_objects.dyn_cart_mri_enc_obj import Dyn2DCartEncObj
from ...networks.efficientnet import EfficientNet, MBConfig
from ...networks.q_cart_mri_primal_dual_nn import QCartPrimalDualNN
from ...networks.unet import UNet
from ...utils.maskedloss import maskedloss
from ...utils.save_animation import save_animation as _save_animation
from ...utils.conjugate_gradient import conj_grad
from ...utils.maskedssim import SSIM

def fillMask(m):
    ret=m.clone()
    t=m[0].cpu().numpy()
    t[0]=1
    t[-1]=1
    t=binary_fill_holes(t)
    ret[0]=torch.as_tensor(t)
    ret[0,0]=0
    ret[0,-1]=0
    return ret

def save_animation(x, *args, **kwargs):
    _save_animation(x.detach().cpu(), *args, **kwargs)


def fit(q, y):
    xp = q.initial_values.broadcast_to((y.shape[0], q.nIn, *y.shape[1:-1])).contiguous().requires_grad_(True)
    optim = torch.optim.LBFGS([xp,], lr=0.5, line_search_fn="strong_wolfe", max_iter=200)

    def closure():
        optim.zero_grad()
        yp = torch.nan_to_num(q(xp))
        loss = torch.nn.functional.mse_loss(yp.abs(), y.abs())
        loss.backward()
        with torch.no_grad():
            xp[:, 0] = xp[:, 0].abs()
        return loss

    optim.step(closure)
    xp = xp.detach()
    xp[:, 0] = xp[:, 0].abs()
    return xp


def save_array(what, filename):
    np.save(filename, np.array(torch.as_tensor(what).detach().cpu()))


parser = argparse.ArgumentParser()
parser.add_argument("valfile", type=str)
parser.add_argument("outpath", type=str)
parser.add_argument("modelfile", type=str, nargs="+")
parser.add_argument("--maskweight", type=float, default=0.0)
parser.add_argument("--const_lambda", action="store_true")
parser.add_argument("--cg_iterations", type=int, default=None)
parser.add_argument("--pdhg_iterations", type=int, default=None)
parser.add_argument("--examples", type=int, default=[20,34,58,97], nargs='+')



# models = ["neu_scalar_onval.pt", "neu_unet.pt", "neu_scalar_ontrain.pt", "neu_unet_lessiterations.pt", "neu_scalarcnn_onval.pt", "neu_scalarcnn_ontrain.pt"]
# models=['grid_val_xy_t.pt']
# args = parser.parse_args(["val2.h5", "grid_test", *models])  # lambdacnn2_unet.pt
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = hdf5DS(args.valfile)
dl = torch.utils.data.DataLoader(ds)

for modelfile in args.modelfile:
    print(f"doing {modelfile}")
    modelname = str(modelfile).split(".")[0]
    model = torch.load(modelfile)
    signalfunction = globals()[model["signalfunction"]]
    q = signalfunction().to(device)
    EncObj = Dyn2DCartEncObj()

    if model["mode"] == "lambda_cnn":
        CNN_block = UNet(**model["netsettings"])
    elif model["mode"] == "lambda_xy_t_cnn":
        CNN_block = EfficientNet(**model["netsettings"])
    else:
        CNN_block = None
    if "opt_lambdas" in model:
        if args.const_lambda:
            losses = np.array(model["test_losses"])
            lambdas = np.array(model["test_lambdas"])
            ix = np.argmin(losses.mean(0))
            const_lam = lambdas.reshape(-1, losses[0].size)[..., ix]
            const_lam = torch.as_tensor(const_lam).to(device=device, dtype=torch.float32)
            additional = [const_lam,] * len(dl)
            modelname += "_constant"
        else:
            additional = model["opt_lambdas"].to(device=device, dtype=torch.float32)
    else:
        additional = [None] * len(dl)

    pdhg = QCartPrimalDualNN(EncObj, CNN_block=CNN_block, nu=model["nu"][-1], mode=model["mode"], low_bound=model["low_bound"], up_bound=model["up_bound"]).to(device=device)
    pdhg.load_state_dict(model["state"], strict=False)

    modifier = ''
    if args.pdhg_iterations is not None:
        pdhg.nu = args.pdhg_iterations
    if args.cg_iterations is not None:
        pdhg.prestep_CG = args.cg_iterations
    
    if args.pdhg_iterations==0:
        outpath = Path(args.outpath) / f'CG_{pdhg.prestep_CG}'
    else:
        outpath = Path(args.outpath) / modelname
    print(f"writing to {outpath}")

    outpath.mkdir(parents=True, exist_ok=True)
    maskedssim=SSIM(1.0,channel=1)
    with h5py.File(outpath / "results.h5", "w") as resultsfile:
        pbar = tqdm(zip(dl, additional))
        for nbatch, (batch, lam) in enumerate(pbar):
            x, y, mask, classes, noise, csm, kmask = (i.to(device=device) for i in batch)
            ku = EncObj.apply_A(y, csm, kmask)
            signalstrength = torch.view_as_real(ku[(kmask.unsqueeze(1) > 0.5).broadcast_to(ku.shape)]).std()
            noisestrength = torch.view_as_real(noise[(kmask.unsqueeze(1) > 0.5).broadcast_to(ku.shape)]).std()
            SNR = signalstrength / noisestrength
            ku = ku + EncObj.apply_mask(noise, kmask)
            yu = EncObj.apply_AH(ku, csm, kmask)
            with torch.no_grad():
                if lam is not None:
                    old = pdhg.set_lambda(lam)

                yTV, lambda_reg = pdhg(yu, ku, kmask, csm)

                if lam is not None:
                    pdhg.set_lambda(old)

            loss = maskedloss(yTV, y, mask.unsqueeze(-1), maskweight=args.maskweight)
            ssim = maskedssim(yTV.abs().moveaxis(-1,0), y.abs().moveaxis(-1,0), fillMask(mask))
            spatial_min, spatial_max = torch.min(lambda_reg[:, 0, ...]).item(), torch.max(lambda_reg[:, 0, ...]).item()
            temporal_min, temporal_max = torch.min(lambda_reg[:, 2, ...]).item(), torch.max(lambda_reg[:, 2, ...]).item()
            xp = fit(q, yTV)
            lossParameter = maskedloss(xp, x, mask.unsqueeze(1), maskweight=args.maskweight)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lossParam": f"{lossParameter:.4f}", "spatial": f"{spatial_min:.4f}/{spatial_max:.4f}", "temp": f"{temporal_min:.4f}/{temporal_max:.4f}"})
            h5append(resultsfile, "loss/image", loss.item())
            h5append(resultsfile, "loss/imagessim", ssim.item())
            h5append(resultsfile, "loss/parameter", lossParameter.item())
            h5append(resultsfile, "parameters/r1/gt", x[:, 1].detach().cpu())
            h5append(resultsfile, "parameters/r1/prediction", xp[:, 1].detach().cpu())
            h5append(resultsfile, "parameters/t1/gt", 1/x[:, 1].detach().cpu())
            h5append(resultsfile, "parameters/t1/prediction", 1/xp[:, 1].detach().cpu())
            h5append(resultsfile, "parameters/absm0/gt", x[:, 0].detach().cpu())
            h5append(resultsfile, "parameters/absm0/prediction", xp[:, 0].detach().cpu())
            h5append(resultsfile, "classes", classes.cpu())
            h5append(resultsfile, "SNR", SNR.item())
            h5append(resultsfile, "NoiseStd", noisestrength.item())

            

            t1 = 1000 / (xp[0, 1]).cpu()
            t1gt = 1000 / (x[0, 1]).cpu()
            classes = batch[3]
            with open(outpath / "rms_t1_over_valDS.txt", "w") as rmsf:
                rmsf.write(f"class,rms_in_ms\n")
                for c, classname in enumerate(values_t1t2pd3T.keys()):
                    m = classes[0] == c
                    l = torch.nn.functional.mse_loss(t1[m], t1gt[m], reduction="sum")
                    h5append(resultsfile, f"classesT1Loss/{classname}/losssum", l.item())
                    h5append(resultsfile, f"classesT1Loss/{classname}/masksum", m.sum().item())
                    ls = np.array(resultsfile["classesT1Loss"][classname]["losssum"])
                    ms = np.array(resultsfile["classesT1Loss"][classname]["masksum"])
                    rms = np.sqrt(ls.sum() / ms.sum())
                    try:
                        resultsfile[f"classesT1Loss/{classname}/rms"] = rms
                    except:
                        resultsfile[f"classesT1Loss/{classname}/rms"][...] = rms
                    rmsf.write(f"{classname},{float(rms):.3f}\n")

            if nbatch in args.examples:
                exoutpath = outpath / str(nbatch)
                exoutpath.mkdir(exist_ok=True)

                save_animation(lambda_reg[0, 0], exoutpath / "lambda_spatial.gif", vmax=0.02, vmin=0.0)
                save_animation(lambda_reg[0, 2], exoutpath / "lambda_temporal.gif", vmax=0.02, vmin=0.0)
                save_animation(yu[0].abs(), exoutpath / "y_zero_filled.gif", cmap="gray")
                save_animation(yTV[0].abs(), exoutpath / "y_TV.gif", cmap="gray")

                save_array(lambda_reg[0, 0], exoutpath / "lambda_spatial.npy")
                save_array(lambda_reg[0, 2], exoutpath / "lambda_temporal.npy")
                save_array(yu[0], exoutpath / "y_zero_filled.npy")
                save_array(yTV[0], exoutpath / "y_TV.npy")

                # T1 Prediction
                t1 = 1000 / (xp[0, 1] * mask[0])
                t1 = t1.detach().cpu()
                plt.matshow(t1, vmin=200, vmax=2500, cmap="gray")
                plt.axis("off")
                plt.colorbar(label="$T_1$  $[ms]$")
                plt.savefig(exoutpath / "xp.png", dpi=144)
                np.save(exoutpath / "xp_t1.npy", np.array(t1))

                # T1 GT
                t1 = 1000 / (x[0, 1] * mask[0])
                t1 = t1.detach().cpu()
                plt.matshow(t1, vmin=200, vmax=2500, cmap="gray")
                plt.axis("off")
                plt.colorbar(label="$T_1$  $[ms]$")
                plt.savefig(exoutpath / "gt.png", dpi=144)
                np.save(exoutpath / "gt_t1.npy", np.array(t1))

                # Y GT
                save_animation(y[0].abs(), exoutpath / "y_gt.gif", cmap="gray")
                save_array(y[0], exoutpath / "y_gt.npy")

                # Zero Filled
                xu = fit(q, yu)
                t1 = 1000 / (xu[0, 1] * mask[0])
                t1 = t1.detach().cpu()
                plt.matshow(t1, vmin=200, vmax=2500, cmap="gray")
                plt.axis("off")
                plt.colorbar(label="$T_1$  $[ms]$")
                plt.savefig(exoutpath / "t1_zerofilled.png", dpi=144)
                np.save(exoutpath / "t1_zerofilled.npy", np.array(t1))

                # cg-sense
                aha = lambda x: pdhg.EncObj.apply_AHA(x, csm, kmask)
                yCG = conj_grad(aha, yu, yu, niter=8)
                xcg = fit(q, yCG)
                t1 = 1000 / (xcg[0, 1] * mask[0])
                t1 = t1.detach().cpu()
                plt.matshow(t1, vmin=200, vmax=2500, cmap="gray")
                plt.axis("off")
                plt.colorbar(label="$T_1$  $[ms]$")
                plt.savefig(exoutpath / "t1_cg.png", dpi=144)
                np.save(exoutpath / "t1_cg.npy", np.array(t1))
                save_animation(yCG[0].abs(), exoutpath / "y_cg.gif", cmap="gray")
                save_array(yCG[0], exoutpath / "y_cg.npy")
