import glob
import nibabel as nib
import numpy as np
from os.path import isfile
import datetime
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import torch
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.nn import MSELoss
from monai.data import Dataset, DataLoader, partition_dataset
from monai.transforms.utility.dictionary import SqueezeDimd
from monai.transforms import Compose, LoadImaged, SqueezeDim, SqueezeDimd, ToTensord, LoadImage, ToTensor, EnsureChannelFirstD, EnsureChannelFirst, Resize, RandBiasFieldd
from monai.utils import set_determinism
from glob import glob
import random

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
        return {'image': image, 'mask': mask}

# Assuming you have a list of paired image and bias file paths
#image_files = ['image1.nii.gz', 'image2.nii.gz']
#mask_files = ['bias_field1.nii.gz', 'bias_field2.nii.gz']

mask_files = (glob('/deneb_disk/macaque_atlas_data/mac_bse_data/data/site-uwo/sub-*/ses-001/anat/sub-*_ses-001_run-1_T1w_mask.nii.gz'))
image_files = [m[:-12]+'.nii.gz' for m in mask_files ]

print('*********************')
print(image_files)
print(mask_files)
print('*********************')

# Define transformations
#transforms = Compose([LoadImaged(keys=['image', 'bias'],image_only=True), AddChanneld(keys=['image', 'bias']), ToTensord(keys=['image', 'bias'])])
#transforms = Compose([LoadImage(image_only=True), Resize(), EnsureChannelFirst(), ToTensor()])


data_dicts = [{"image": image, "mask": mask} for image, mask in zip(image_files, mask_files)]

random.seed(11)

#random.shuffle(data_dicts)
num_files = len(data_dicts)
num_train_files = round(0.8 * num_files)
train_files = data_dicts[:num_train_files]
val_files = data_dicts[num_train_files:]
print("total num files:", len(data_dicts))
print("num training files:", len(train_files))
print("num validation files:", len(val_files))

from monai.transforms import Compose, Resized, RandBiasFieldd, ScaleIntensityd,LoadImaged, EnsureChannelFirstd, RandAffined, ToTensord,LoadImage,ToTensor,EnsureChannelFirstD,EnsureChannelFirst, Resize, RandBiasFieldd
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    pad_list_data_collate,
    TestTimeAugmentation,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keys = ["image", "mask"]

train_transforms = Compose([
    LoadImaged(keys,image_only=True),
    EnsureChannelFirstd(keys),
    ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
    Resized(
            keys,
            spatial_size=(64, 64, 64),
            mode='trilinear',
    ),
    RandAffined(
            keys,
            prob=0.5,
            #rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
            rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
            #translate_range=(5,5,5),
            translate_range=(15,15,15),
            scale_range=(0.3,0.3,0.3),
            shear_range=(.1,.1,.1,.1,.1,.1),
            padding_mode=("zeros","reflection"),
        ),
    RandBiasFieldd(keys="image",prob=0.5, coeff_range=(-1,1), degree=5),
])


val_transforms = Compose([
    LoadImaged(keys,image_only=True),
    EnsureChannelFirstd(keys),
    ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
    Resized(
            keys,
            spatial_size=(64, 64, 64),
            mode='trilinear',
    ),
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=16, num_workers=10, collate_fn=pad_list_data_collate)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=10, collate_fn=pad_list_data_collate)

batch = next(iter(train_loader))

i=0
for batch in train_loader:
    i += 1
    if i>20:
        break

    #for j in range(4):
    #    print(j)
    #    plt.subplot(121)
    #    plt.imshow(batch['image'][j,0,:,32,:],cmap='gray',vmin=0,vmax=1)
    #    plt.subplot(122)
    #    plt.imshow(batch['mask'][j,0,:,32,:],cmap='gray',vmin=-10,vmax=10)

    #    plt.show()

# Create dataset and data loader
#dataset = BSEDataset(image_files, mask_files, transform=transforms)


# Split the dataset into training and validation sets


# Create data loaders for training and validation
#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define the UNet model and optimizer

# Specify spatial_dims and strides for 3D data
spatial_dims = 3
strides = (1, 1, 1, 1)

model = UNet(
    spatial_dims=spatial_dims,
    in_channels=1,  # Adjust based on your data
    out_channels=1, # Adjust based on your data
    channels=(2,8,8,16,32),#(16, 64, 64, 128, 256),
    strides=strides,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the Dice loss
loss_function = MSELoss() # DiceLoss(sigmoid=True)

# Training loop
num_epochs = 20002
save_interval = 500

train_loss_epoch = np.zeros(num_epochs)
val_loss_epoch = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:

        inputs, masks = batch['image'].to(device), batch['mask'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, masks)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_train_loss / len(train_files)}")
    train_loss_epoch[epoch] = total_train_loss / len(train_files)


    # Validation loop
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
            total_val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss / len(val_files)}")
    val_loss_epoch[epoch] = total_val_loss / len(val_files)


    if epoch % save_interval == 0:

        current_datetime = datetime.datetime.now()

        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Create a filename with the formatted date and time
        filename = f"models/macbse_model_{formatted_datetime}_epoch_{epoch}.pth"

        # Save the trained model
        torch.save(model.state_dict(), filename)
        filename = f"models/macbse_loss_{formatted_datetime}_epoch_{epoch}.npz"

        np.savez(filename,val_loss_epoch=val_loss_epoch,train_loss_epoch=train_loss_epoch)

print('Training is done!')

