import torch
import numpy as np
import argparse
import random
from ...encoding_objects.dyn_cart_mri_enc_obj import Dyn2DCartEncObj
from ...data.qmri.h5 import to_hdf5
from ...data.qmri import QCartUSDataset, T1Inversion, T1Saturation

parser = argparse.ArgumentParser()
parser.add_argument("datapath", type=str)
parser.add_argument("outpath", type=str)
parser.add_argument("--signalfunction", default="T1Inversion", type=str)
parser.add_argument("--imagesize", default=192, type=int)
parser.add_argument("--acc_factor", default=6, type=int, nargs="+")
parser.add_argument("--noise_strength", default=[0.005, 0.05], type=float, nargs="+")
parser.add_argument("--Ncoils", default=8, type=int)
args = parser.parse_args()
signalfunction = globals()[args.signalfunction]

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

ds = QCartUSDataset(signalfunction(), args.datapath, acceleration=args.acc_factor, Ncoils=args.Ncoils, Npx=args.imagesize, augment=False, random_phase=True, noise_strength=args.noise_strength)
to_hdf5(ds, args.outpath)
