Dynamic Image Denoising Example
===============================


A simple training script can be found in `/scripts/dyn_img/train_dyn_img.py`.

For testing purposes, the number of scenes for the training and the validation dataset was limited to one.
The original selection of scenes that was used to train the model for the paper can be found at the top of the script.
Also note, that the number of epochs is set to 2 for testing purposes.


Before running the training script, please ensure that the MOT17Det dataset was downloaded successfully using the script `/data/dyn_img/download_mot_data.py`.
The training example assumes that the data has been downloaded to a `/tmp` folder, located in the same directory as the download script.
This can be achieved by the following command.
```
python download_mot_data.py tmp
```

Make sure to edit the `data_path` argument of the dataset in the training script in case you chose a different download destination (when using the command from above it is `"../../data/dyn_img/tmp"`). The example training script can be executed by the following command.
```
python train_dyn_img.py
```

The terminal output should be as follows:

```
loading image id 02, 0/1
extracting patches of shape [192, 192, 32]; strides [192, 192, 16]
loading image id 11, 0/1
extracting patches of shape [192, 192, 32]; strides [192, 192, 16]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [06:22<00:00,  1.06s/it]
TRAINING LOSS:  0.006664330658774513
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 360/360 [06:20<00:00,  1.06s/it]
TRAINING LOSS:  0.0061657700493621325
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 550/550 [01:34<00:00,  5.80it/s]
VALIDATION LOSS:  0.0061657700493621325
```