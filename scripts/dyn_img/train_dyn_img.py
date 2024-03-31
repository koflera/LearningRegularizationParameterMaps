# %%
import sys
import os

import torch
from torch.utils.data import WeightedRandomSampler

from helper_functions import train_epoch, validate_epoch

sys.path.append("../../")
from data.dyn_img.dataset import DynamicImageDenoisingDataset
from networks.dyn_img_primal_dual_nn import DynamicImagePrimalDualNN
from networks.unet import UNet


TRAINING = [2, 4, 5, 9, 10]
VALIDATION = [11, 13]

DEVICE = torch.device("cuda:0")

# %%
# Dynamic Image Denoising Dataset

# Make sure that the dataset was downloaded successfully
# The script to download the MOT17Det dataset can be found here: /data/dyn_img/download_mot_data.py
# The data samples can be created with different scaling factors.
# Make sure to set extract_data to True when loading the dataset for the first time to create the dynamic images.
# Once the data for a specific scaling factor has been created the flag can be set to False.
dataset_train = DynamicImageDenoisingDataset(
    data_path="../../data/dyn_img/tmp",
    # ids=TRAINING,     # paper
    ids=[2],            # testing
    scale_factor=0.5,
    sigma=[0.1, 0.3],
    strides=[192, 192, 16],
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
    # extract_data=False,
    device=DEVICE
)

# Create training dataloader
sampler = WeightedRandomSampler(dataset_train.samples_weights, len(dataset_train.samples_weights))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, sampler=sampler)

# Validation dataset (see note above)
dataset_valid = DynamicImageDenoisingDataset(
    data_path="../../data/dyn_img/tmp",
    # ids=VALIDATION,   # paper
    ids=[11],           # testing
    scale_factor=0.5,
    sigma=[0.1, 0.3],
    strides=[192, 192, 16],
    patches_size=[192, 192, 32],
    # (!) Make sure to set the following flag to True when loading the dataset for the first time.
    extract_data=True,
    # extract_data=False,
    device=DEVICE
)

# Create validation dataloader 
sampler = WeightedRandomSampler(dataset_valid.samples_weights, len(dataset_valid.samples_weights))
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, sampler=sampler)


# %%
# Define CNN block and PDHG-method
unet = UNet(dim=3, n_ch_in=1).to(DEVICE)

# Constrct primal-dual operator with nn
pdhg = DynamicImagePrimalDualNN(
    cnn_block=unet, 
    T=128,
    phase="training",
    up_bound=0.5,
    # Select mode:
    mode="lambda_cnn",
    # mode="lambda_xy_t",
    # mode="lambda_xyt",
).to(DEVICE)

optimizer = torch.optim.Adam(pdhg.parameters(), lr=1e-4)
loss_function = torch.nn.MSELoss()

num_epochs = 2          # testing
# num_epochs = 100      # paper

model_states_dir = "./tmp/states"
os.makedirs(model_states_dir, exist_ok=True)

for epoch in range(num_epochs):

    # Model training
    pdhg.train(True)
    training_loss = train_epoch(pdhg, dataloader_train, optimizer, loss_function)
    pdhg.train(False)
    print("TRAINING LOSS: ", training_loss)

    if (epoch+1) % 2 == 0:

        with torch.no_grad():

            # Model validation
            validation_loss = validate_epoch(pdhg, dataloader_valid, loss_function)
            print("VALIDATION LOSS: ", validation_loss)
            torch.save(pdhg.state_dict(), f"{model_states_dir}/epoch_{str(epoch).zfill(3)}.pt")

    torch.cuda.empty_cache()

# Save the entire model
torch.save(pdhg, f"./tmp/model.pt")

# %%
