import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet #SwinUNETR, unet
from torch.nn import MSELoss
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, RandBiasFieldd
from monai.utils import set_determinism
from glob import glob
import random
import os

from monai.transforms import (
    Compose,
    HistogramNormalizeD,
    Resized,
    RandBiasFieldd,
    ScaleIntensityd,
    LoadImaged,
    EnsureChannelFirstd,
    RandAffined,
    RandBiasFieldd,
)
import pandas as pd
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    pad_list_data_collate,
)

from monai.transforms import (
    RandAdjustContrastd,
    RandGaussianNoised,
    HistogramNormalizeD,
)

# Set random seed for reproducibility
set_determinism(seed=0)


# Define your dataset and data loader
class BSEDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.transform(self.image_files[idx])
        mask = self.transform(self.mask_files[idx])
        return {"image": image, "mask": mask}


# Assuming you have a list of paired image and bias file paths
# image_files = ['image1.nii.gz', 'image2.nii.gz']
# mask_files = ['bias_field1.nii.gz', 'bias_field2.nii.gz']

# mask_files = (glob('/deneb_disk/macaque_atlas_data/mac_bse_data/data/site-uwo/sub-*/ses-001/anat/sub-*_ses-001_run-1_T1w_mask.nii.gz'))


# check if the directory exists

if os.path.exists("/deneb_disk/mac_bse_data/data"):
    data_dir = "/deneb_disk/mac_bse_data/data"
else:
    data_dir = "/scratch1/ajoshi/mac_bse_data/data"


mask_files = glob(f"{data_dir}/sub*_mask.nii.gz")

image_files = [m[:-12] + ".nii.gz" for m in mask_files]


df = pd.DataFrame({"Image Files": image_files, "Mask Files": mask_files})
print(df)

# print a list of image files


#
# Define transformations
# transforms = Compose([LoadImaged(keys=['image', 'bias'],image_only=True), AddChanneld(keys=['image', 'bias']), ToTensord(keys=['image', 'bias'])])
# transforms = Compose([LoadImage(image_only=True), Resize(), EnsureChannelFirst(), ToTensor()])


data_dicts = [
    {"image": image, "mask": mask} for image, mask in zip(image_files, mask_files)
]

random.seed(11)

# random.shuffle(data_dicts)
num_files = len(data_dicts)
num_train_files = round(0.8 * num_files)
train_files = data_dicts[:num_train_files]
val_files = data_dicts[num_train_files:]
print("total num files:", len(data_dicts))
print("num training files:", len(train_files))
print("num validation files:", len(val_files))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = ["image", "mask"]

train_transforms = Compose(
    [
        LoadImaged(keys, image_only=True),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys="image", minv=0.0, maxv=255.0),
        Resized(
            keys,
            spatial_size=(64, 64, 64),
            mode=("trilinear","nearest"),
        ),
        RandAffined(
            keys,
            prob=0.5,
            # rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
            rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
            # translate_range=(5,5,5),
            translate_range=(15, 15, 15),
            scale_range=(0.3, 0.3, 0.3),
            shear_range=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            padding_mode=("zeros", "zeros"),
            mode=("trilinear","nearest"),
        ),
        RandBiasFieldd(keys="image", prob=0.5, coeff_range=(0, 0.5)),
        RandAdjustContrastd(
            keys="image", prob=0.5, invert_image=True, gamma=(0.5, 2.0)
        ),
        RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=50),
        # RandGaussianSmoothd(keys="image",prob=0.5,sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5)),
        #HistogramNormalizeD(keys="image", num_bins=255),

    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys, image_only=True),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys="image", minv=0.0, maxv=255.0),
        Resized(
            keys,
            spatial_size=(64, 64, 64),
            mode="trilinear",
        ),
        #HistogramNormalizeD(keys="image", num_bins=255),
    ]
)

train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4
)
train_loader = DataLoader(
    train_ds, batch_size=8, num_workers=10, collate_fn=pad_list_data_collate
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, num_workers=10, collate_fn=pad_list_data_collate
)

batch = next(iter(train_loader))

for j in range(8):
    plt.subplot(121)
    plt.imshow(batch["image"][j, 0, :, 32, :], cmap="gray", vmin=0, vmax=255)
    plt.subplot(122)
    plt.imshow(batch["mask"][j, 0, :, 32, :], cmap="gray", vmin=0, vmax=1)
    plt.show()


batch = next(iter(val_loader))

for j in range(1):
    plt.subplot(121)
    plt.imshow(batch["image"][j, 0, :, 32, :], cmap="gray", vmin=0, vmax=255)
    plt.subplot(122)
    plt.imshow(batch["mask"][j, 0, :, 32, :], cmap="gray", vmin=0, vmax=1)
    plt.show()


# Define the UNet model and optimizer

# Specify spatial_dims and strides for 3D data
spatial_dims = 3
strides = (1, 1, 1, 1)

"""
model = SwinUNETR(
    img_size=(64, 64, 64),
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
).to(device)
"""
model = UNet(
    spatial_dims=spatial_dims,
    in_channels=1,  # Adjust based on your data
    out_channels=1,  # Adjust based on your data
    channels= (16, 64, 64, 128, 256), #(2, 8, 8, 16, 32), #(16, 64, 64, 128, 256),
    strides=strides,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the Dice loss
loss_function = DiceLoss(sigmoid=True, reduction="sum") #MSELoss()  # 

# Training loop
num_epochs = 20002
save_interval = 500

train_loss_epoch = np.zeros(num_epochs)
val_loss_epoch = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:

        inputs, masks = batch["image"].to(device), batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, masks)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_train_loss / len(train_files)}"
    )
    train_loss_epoch[epoch] = total_train_loss / len(train_files)

    # Validation loop
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
            total_val_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss / len(val_files)}"
    )
    val_loss_epoch[epoch] = total_val_loss / len(val_files)

    if epoch % save_interval == 0:

        current_datetime = datetime.datetime.now()

        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Create a filename with the formatted date and time
        filename = f"models/macbse_aug_model_{formatted_datetime}_epoch_{epoch}.pth"

        # Save the trained model
        torch.save(model.state_dict(), filename)
        filename = f"models/macbse_aug_loss_{formatted_datetime}_epoch_{epoch}.npz"

        np.savez(
            filename, val_loss_epoch=val_loss_epoch, train_loss_epoch=train_loss_epoch
        )

print("Training is done!")
