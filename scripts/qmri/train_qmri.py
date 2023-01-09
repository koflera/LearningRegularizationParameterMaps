import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc

from ...encoding_objects.q_cart_mri_enc_obj import Q2DCartEncObj
from ...networks.q_cart_mri_primal_dual_nn import QCartPrimalDualNN
from ...data.qmri import QCartUSDataset, T1Inversion, T1Saturation
from ...data.qmri.h5 import hdf5DS
from ...networks.unet import UNet
from ...networks.efficientnet import EfficientNet, MBConfig
from ...utils.warmup import WarmupLR
from ...utils.maskedloss import maskedloss
from ...utils.save_animation import save_animation


def mode(modestr):
    validmodes = (
        "lambda_xyt",
        "lambda_xy_t",
        "lambda_xy_t_cnn",
        "lambda_cnn",
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
parser.add_argument("--mode", default="lambda_cnn", type=mode)
parser.add_argument("--E", default=3, type=int)
parser.add_argument("--C", default=2, type=int)
parser.add_argument("--K", default=8, type=int)
parser.add_argument("--low_bound", default=0, type=float)
parser.add_argument("--up_bound", default=0.0, type=float)
parser.add_argument("--Nepochs", default=10, type=int)
parser.add_argument("--batchsize", default=1, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--maskweight", default=0.2, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--warmup", default=1, type=float)
parser.add_argument("--gradaccum", default=1, type=int)
parser.add_argument("--neptune", default=False, action="store_true")
parser.add_argument("--savefile", default="./model.pt", type=str)
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
dl = torch.utils.data.DataLoader(ds, batch_size=args.batchsize, shuffle=True, pin_memory=True)

EncObj = Q2DCartEncObj()

if args.mode == "lambda_cnn":
    netsettings = dict(dim=3, n_enc_stages=args.E, n_convs_per_stage=args.C, n_filters=args.K, res_connection=False, bias=True, padding_mode="reflect")
    CNN_block = UNet(**netsettings)
    with torch.no_grad():

        def bias_to_zero(m):
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.fill_(0)

        CNN_block.apply(bias_to_zero)
        CNN_block.c1x1.bias.fill_(-1.0)  # force initial lambdas to be closer to lower bound / zero
elif args.mode == "lambda_xy_t_cnn":
    blocksettings = (
        MBConfig(fused=True, expand_ratio=1, kernel=3, stride=1, input_channels=16, output_channels=16, num_layers=2),
        MBConfig(fused=True, expand_ratio=4, kernel=(3, 3, 1), stride=(2, 2, 1), input_channels=16, output_channels=32, num_layers=3),
        MBConfig(fused=False, expand_ratio=4, kernel=3, stride=2, input_channels=32, output_channels=48, num_layers=3),
        MBConfig(fused=False, expand_ratio=4, kernel=(3, 3, 1), stride=(2, 2, 1), input_channels=48, output_channels=64, num_layers=4),
        MBConfig(fused=False, expand_ratio=4, kernel=3, stride=2, input_channels=64, output_channels=128, num_layers=6),
    )
    netsettings = dict(dim=3, input_channels=2, output_values=2, settings=blocksettings)
    CNN_block = EfficientNet(**netsettings)
    with torch.no_grad():
        CNN_block.classifier[-1].bias.fill_(0.0)
else:
    netsettings = dict()
    CNN_block = None

pdhg = QCartPrimalDualNN(EncObj, CNN_block=CNN_block, nu=args.nu[0], mode=args.mode, low_bound=args.low_bound, up_bound=None if args.up_bound == 0.0 else args.up_bound).to(device=device)


optim = torch.optim.AdamW(pdhg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = WarmupLR(torch.optim.lr_scheduler.CosineAnnealingLR(optim, (args.Nepochs - args.warmup) * len(dl), eta_min=args.lr / 30, verbose=False), init_lr=args.lr / 30, num_warmup=args.warmup * len(dl))

for epoch in tqdm(range(args.Nepochs)):
    gc.collect()
    pbar = tqdm(dl)
    n = int(np.round(np.linspace(0, len(args.nu) - 1, args.Nepochs))[epoch])
    pdhg.nu = args.nu[n]
    for batchid, batch in enumerate(pbar):
        x, y, mask, classes, noise, csm, kmask = (i.to(device=device) for i in batch)
        ku = EncObj.apply_A(y, csm, kmask) + EncObj.apply_mask(noise, kmask)
        xu = EncObj.apply_AH(ku, csm, kmask)
        xTV, lambda_reg = pdhg(xu, ku, kmask, csm)
        loss = maskedloss(xTV, y, mask.unsqueeze(-1), maskweight=args.maskweight) / float(args.gradaccum)
        loss.backward()
        spatial_min, spatial_max = torch.min(lambda_reg[:, 0, ...]).item(), torch.max(lambda_reg[:, 0, ...]).item()
        temporal_min, temporal_max = torch.min(lambda_reg[:, 2, ...]).item(), torch.max(lambda_reg[:, 2, ...]).item()
        if args.neptune:
            run["loss"].log(loss.item())
            run["lambda/spatial/min"].log(spatial_min)
            run["lambda/spatial/max"].log(spatial_max)
            run["lambda/temporal/min"].log(temporal_min)
            run["lambda/temporal/max"].log(temporal_max)
            run["lr"].log(optim.param_groups[0]["lr"])
            run["nu"].log(pdhg.nu)
            run["sigma"].log(pdhg.sigma)
            run["tau"].log(pdhg.tau)
            run["theta"].log(pdhg.theta)
            if batchid % 100 == 0:
                spatial = lambda_reg[0, 0].detach().cpu()
                temporal = lambda_reg[0, 2].detach().cpu()
                img = xTV[0].detach().cpu()
                run["lambda/spatial/gif"].log(neptune.types.File(save_animation(spatial)))
                run["lambda/temporal/gif"].log(neptune.types.File(save_animation(temporal)))
                run["images/xTV/abs"].log(neptune.types.File(save_animation(img.abs())))
                run["images/xTV/real"].log(neptune.types.File(save_animation(img.real)))
                run["images/xTV/imag"].log(neptune.types.File(save_animation(img.real)))

        sched.step()
        if (batchid + 1) % args.gradaccum == 0:
            optim.step()
            optim.zero_grad(True)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "spatial": f"{spatial_min:.2f}/{spatial_max:.2f}", "temp": f"{temporal_min:.2f}/{temporal_max:.2f}"})

pdhg.cpu()
tosave = dict(state=pdhg.state_dict(), **vars(args), netsettings=netsettings)
torch.save(tosave, args.savefile)
print(f"saved to {args.savefile}")

if args.neptune:
    run.stop()
