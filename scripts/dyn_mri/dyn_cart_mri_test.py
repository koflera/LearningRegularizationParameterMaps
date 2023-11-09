# %%
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append("../../")
from networks.dyn_cart_mri_primal_dual_nn import DynCartPrimalDualNN
from networks.unet import UNet
from encoding_objects.dyn_cart_mri_enc_obj import Dyn2DCartEncObj
from utils.mrisensesim import generate_2d_csmaps
from utils.cartesian_mask_funcs import cine_cartesian_mask
from utils.conjugate_gradient import conj_grad

# define the dynamic mri encoding object which contains the forward
# and the adjoint operators
EncObj = Dyn2DCartEncObj()

# choose lambda mode; i.e. lambda_xyt, lambda_xy_t or lambda_cnn
# lambda_mode = "lambda_cnn"  # spatio-temporal parameter map
# lambda_mode = "lambda_xy_t"  # two scalara parameters (space and time)
lambda_mode = "lambda_xyt"  # same scalar for all directional derivatives

if lambda_mode == "lambda_cnn":
    # hyper parameters of the U-Net
    dim = 3  # 3D U-Net

    # nr of encoding stages, conv layer per stage, initial number of filters
    E, C, K = 3, 2, 8  # could be in principle changed and further investigated
    # define the CNN-block used to estimate the spatio-temporal parameter maps
    cnn = UNet(
        dim,
        n_ch_in=2,
        n_ch_out=2,
        n_enc_stages=E,
        n_convs_per_stage=C,
        n_filters=K,
        bias=False,
        res_connection=True,
    )

    # load the weights of the U-Net which were obtained by training with
    # T_train unrolled iterations (see the paper for details)
    T_train = 256
    pname = "../../networks/pre_trained_models/dyn_mri/"
    fname = "unet_T_train{}.pth".format(T_train)
    cnn.load_state_dict(torch.load(pname + fname))

elif lambda_mode == "lambda_xyt":
    lambda_reg = torch.nn.Parameter(torch.tensor([-4.1973]))
    cnn = None

elif lambda_mode == "lambda_xy_t":
    lambda_reg = torch.nn.Parameter(torch.tensor([-5.3583, -3.2171]))
    cnn = None

# define the unrolled PDGH with T iterations
T = 256
pdhg = DynCartPrimalDualNN(EncObj, CNN_block=cnn, T=T, mode=lambda_mode)

if lambda_mode in ["lambda_xyt", "lambda_xy_t"]:
    pdhg.lambda_reg = lambda_reg

# load ground-truth image
x_true = torch.load("../../data/dyn_mri/xtrue.pt")
im_size = tuple(x_true.shape[1:])
im_size_xy = tuple(im_size[:2][::-1])
# generate coil sensitivity maps
ncoils = 12
csm = generate_2d_csmaps(im_size_xy, ncoils)

# generate an undersampling maks
R = 6
mask = cine_cartesian_mask(im_size, R).unsqueeze(0)

# generate undersampled k-space data and add noise
sigma = 0.05
y = EncObj.apply_A(x_true, csm, mask)
mb, Nc, Nx, Ny, Nt = y.shape
y = y + mask.unsqueeze(1) * sigma * y.abs().mean() * torch.randn(y.shape)

# get initial reconstruction
xu = EncObj.apply_AH(y, csm, mask)

if torch.cuda.is_available():
    y = y.cuda()
    mask = mask.cuda()
    csm = csm.cuda()
    xu = xu.cuda()
    pdhg = pdhg.cuda()

with torch.no_grad():
    # approximately solve the normal equations  A^H A x = A^H y, where x0=A^H y
    # for better initialization
    AHA = lambda x: EncObj.apply_AHA(x, csm, mask)
    AHy = xu.clone()
    xneq = conj_grad(AHA, AHy, AHy, niter=8)

    if lambda_mode == "lambda_cnn":
        # estimate lambda_cnn from x0
        lambda_cnn = pdhg.get_lambda_cnn(xneq)
    else:
        lambda_cnn = None

    # reconstruct image using the pdhg method and the estimated parameter-maps
    xtv = pdhg(y, xneq, mask, csm, lambda_map=lambda_cnn)

# plot the reconstruction results
figsize = 4
fig, ax = plt.subplots(2, 4, figsize=(figsize * 4, figsize * 2))
plt.subplots_adjust(wspace=0.05, hspace=-0.4)
arrs_list = [xu.cpu(), xneq.cpu(), xtv.cpu(), x_true.cpu()]
errs_list = [arr - x_true.cpu() for arr in arrs_list]
titles_list = ["Zero-Filed", "Normal Eqs", "PDHG", "Target"]
cmap_img = plt.cm.Greys_r
cmap_err = plt.cm.viridis
clim = [0, 16]
k = 0
for arr, err, title in zip(arrs_list, errs_list, titles_list):
    if title != "Target":
        mse = F.mse_loss(torch.view_as_real(arr), torch.view_as_real(x_true))
        title += "  MSE: {}".format(round(mse.item(), 3))
    ax[0, k].set_title(title)
    ax[0, k].imshow(arr.abs()[0, ..., 5], cmap=cmap_img, clim=clim)
    ax[1, k].imshow(3 * err.abs()[0, ..., 5], cmap=cmap_err, clim=clim)
    k += 1
plt.setp(ax, xticks=[], yticks=[])

if lambda_mode == "lambda_cnn":
    lambda_cnn = lambda_cnn.cpu()
    # plot the obtained spatio-temporal regularization parameter-maps
    fig, ax = plt.subplots(1, 2, figsize=(figsize * 2, figsize * 1))
    cmap = plt.cm.inferno
    clim = [0, 0.25]
    plt.subplots_adjust(wspace=0.05, hspace=-0.4)
    ax[0].imshow(lambda_cnn[0, 0, ..., 5], cmap=cmap, clim=clim)
    ax[1].imshow(lambda_cnn[0, 2, ..., 5], cmap=cmap, clim=clim)
    ax[0].set_title("Spatial Lambda-Map (x-/y-directions)")
    ax[1].set_title("Temporal Lambda-Map (t-direction)")
    plt.setp(ax, xticks=[], yticks=[])
