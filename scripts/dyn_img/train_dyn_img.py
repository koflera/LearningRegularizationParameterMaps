# %%
import sys

import torch
from torch.utils.data import WeightedRandomSampler

sys.path.append("../../")
from data.dyn_img.dataset import DynamicImageDenoisingDataset
from networks.dyn_img_primal_dual_nn import DynamicImagePrimalDualNN
from networks.unet import UNet


TRAINING = [2, 4, 5, 9, 10]
VALIDATION = [11, 13]

# %%
# Dynamic Image Denoising Dataset

# Make sure that the dataset was downloaded successfully
# The script to download the MOT17Det dataset can be found here: /data/dyn_img/download_mot_data.py
# The data samples can be created with different scaling factors.
# Make sure to set extract_data to True when loading the dataset for the first time to create the dynamic images.
# Once the data for a specific scaling factor has been created the flag can be set to False.
dataset_train = DynamicImageDenoisingDataset(
    data_path="../../data/dyn_img/test",
    ids=TRAINING,
    scale_factor=0.5,
    sigma=[0.1, 0.3],
    strides=[192, 192, 16],
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
)

# Create training dataloader
sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, sampler=sampler)

# Validation dataset (see note above)
dataset_valid = DynamicImageDenoisingDataset(
    data_path="../../data/dyn_img/test",
    ids=VALIDATION,
    scale_factor=0.5,
    sigma=[0.1, 0.3],
    strides=[192, 192, 16],
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
)

# Create validation dataloader 
sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=sampler)


# %%

# Define CNN block and PDHG-method
unet = UNet(dim=3, n_ch_in=1)

pdhg = DynamicImagePrimalDualNN(
    cnn_block=unet, 
    T=128,
    data_mode="real",
    phase="training",
    up_bound=0.5,
    
    # Select mode:
    # mode="lambda_cnn",
    # mode="lambda_xy_t",
    mode="lambda_xyt",
).cuda()