
#!/usr/bin/env python
import os
from sys import prefix
import nibabel as nib
from macbse import macbse

import SimpleITK as sitk


# Define paths
bse_model = "/home/ajoshi/macbse_aug_model_2024-03-07_23-05-42_epoch_8000.pth" #"models/bias_field_correction_model_2024-03-02_22-29-46_epoch_9000.pth"
bse_model = "/home/ajoshi/macbse_aug_model_2024-03-14_16-16-03_epoch_1000.pth" #macbse_aug_model_2024-03-08_10-46-25_epoch_16500.pth" #"models/bias_field_correction_model_2024-03-02_22-29-46_epoch_9000.pth"
#bse_model = "/home/ajoshi/macbse_aug_model_2024-03-13_14-16-46_epoch_20000.pth"

#prefix = "/home/ajoshi/Downloads/temp/ONPRC18_LargeFOV_T1W_head"
prefix = "data/sub-032196_ses-001_run-1_T1w"
#"/deneb_disk/macaque_atlas_data/site-uwmadison_part1/sub-1002/anat/sub-1002_T1w" #

mri = f"{prefix}.nii.gz"

bseout = f"{prefix}.bse.nii.gz"
bfcout = f"{prefix}.bfc.nii.gz"
biasfield = f"{prefix}.bias.nii.gz"
maskfile = f"{prefix}.mask.nii.gz"
cerebrum_maskfile = f"{prefix}.cerebrum.mask.nii.gz"
pvcfile = f"{prefix}.pvc.frac.nii.gz"
pvc_label_file = f"{prefix}.pvc.label.nii.gz"
warped_air_atlas = f"{prefix}.warped.airatlas.nii.gz"
warped_air_labels = f"{prefix}.hemi.label.nii.gz"
air_atlas = "/deneb_disk/macaque_atlas_data/macaque_hemi_atlas/NMT_v2.1_sym_05mm.bfc.nii.gz"
#"/home/ajoshi/Downloads/VERVET/brainsuite/VALiDATe12-airatlas/VALiDATe12-t1.airatlas.nii.gz"
air_atlas_labels = "/deneb_disk/macaque_atlas_data/macaque_hemi_atlas/NMT_v2.1_sym_05mm.edit.hemi.label.nii.gz"
#"/home/ajoshi/Downloads/VERVET/brainsuite/VALiDATe12-airatlas/VALiDATe12-t1.airatlas.label.nii.gz"

reg_mat = f"{prefix}.airatlas.mat"


if not os.path.exists(bseout):
    macbse(mri, bseout, bse_model, maskfile, device="cuda")


# use SImpleITK to perform bias field correction
# Read the input image
inputImage = sitk.ReadImage(bseout)

# Set up for processing
maskImage = sitk.ReadImage(maskfile)
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)




# Apply the N4BiasFieldCorrection filter
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([50] * 3)
corrector.SetConvergenceThreshold(1e-6)
corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)

# Execute the filter
outputImage = corrector.Execute(inputImage, maskImage)
log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
bias_field = sitk.Exp(log_bias_field)

# Write the result
sitk.WriteImage(outputImage, bfcout)
sitk.WriteImage(log_bias_field, biasfield)




# do tissue classification in WM GM and CSF. Use FSL's FAST from the command line
# fsl5.0-fast -t 1 -n 3 -g -o sub-032196_ses-001_run-1_T1w.bfc.nii.gz sub-032196_ses-001_run-1_T1w.bfc.nii.gz

# do tissue classification in WM GM and CSF. Use FSL's FAST from the command line
os.system(f"fast -t 1 -n 3 -g -o {prefix} {bfcout}")




# Generate Isosurface from the white matter probability map
import numpy as np
import nibabel as nib

# Load the tissue probability maps
gm = nib.load(f"{prefix}_pve_1.nii.gz").get_fdata()
wm = nib.load(f"{prefix}_pve_2.nii.gz").get_fdata()
csf = nib.load(f"{prefix}_pve_0.nii.gz").get_fdata()

affine = nib.load(f"{prefix}_pve_0.nii.gz").affine

csf_msk = np.double(csf > 0)
gm_msk = np.double(gm > 0)
wm_msk = np.double(wm > 0)

pvc_frac = 1 * csf_msk * (gm_msk == 0)
pvc_frac += (wm_msk == 0) * gm_msk * (1 + gm)
pvc_frac += wm_msk * (2 + wm)


nib.save(nib.Nifti1Image(np.float32(pvc_frac), affine=affine), pvcfile)

nib.save(nib.Nifti1Image(np.uint8(pvc_frac), affine=affine), pvc_label_file)



"""
cmd = f"flirt -in {air_atlas} -ref {bfcout} -out {warped_air_atlas} -omat {reg_mat}"
os.system(cmd)

cmd = f"flirt -in {air_atlas_labels} -ref {bfcout} -out {warped_air_labels} -applyxfm -init {reg_mat} -interp nearestneighbour"
os.system(cmd)

v = nib.load(warped_air_labels).get_fdata()
v = np.uint8(v)
nib.save(nib.Nifti1Image(v, affine=affine), warped_air_labels)


v = nib.load(warped_air_labels).get_fdata()
m = nib.load(maskfile).get_fdata()
m[(v == 3) | (v == 4)] = 0
m = m > 0.5
nib.save(nib.Nifti1Image(255 * np.uint8(m), affine=affine), cerebrum_maskfile)
"""


cmd = f"./cortical_extraction_macaque.sh {prefix}"
os.system(cmd)
