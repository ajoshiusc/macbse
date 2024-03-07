
#!/usr/bin/env python
import os
import nibabel as nib
from macbse import macbse

import SimpleITK as sitk


# Define paths
mri = "sub-032196_ses-001_run-1_T1w.nii.gz"
bseout = "test3.bse.nii.gz"
bfcout = "test3.bfc.nii.gz"
biasfield = "test3.bias.nii.gz"
bse_model = 'models/bias_field_correction_model_2024-03-02_22-29-46_epoch_9000.pth'
maskfile = "sub-032196_ses-001_run-1_T1w.mask.nii.gz"
pvcfile = "sub-032196_ses-001_run-1_T1w.pvc.nii.gz"



macbse(mri, bseout, bse_model,maskfile,device='cuda')


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
os.system(f"fast -t 1 -n 3 -g -o {mri} {bfcout}")




# Generate Isosurface from the white matter probability map
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib

# Load the tissue probability maps
gm = nib.load("sub-032196_ses-001_run-1_T1w.bfc_pve_1.nii.gz").get_fdata()
wm = nib.load("sub-032196_ses-001_run-1_T1w.bfc_pve_2.nii.gz").get_fdata()
csf = nib.load("sub-032196_ses-001_run-1_T1w.bfc_pve_0.nii.gz").get_fdata()

# Threshold the tissue probability maps
threshold = 0.5
#gm[gm < threshold] = 0
#wm[wm < threshold] = 0
#csf[csf < threshold] = 0

# Generate isosurfaces
verts_gm, faces_gm, _, _ = measure.marching_cubes(gm, level=0.5)
verts_wm, faces_wm, _, _ = measure.marching_cubes(wm, level=0.5)
verts_csf, faces_csf, _, _ = measure.marching_cubes(csf, level=0.5)


# Plot the isosurfaces
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of trisurf objects
mesh_gm = Poly3DCollection(verts_gm[faces_gm], alpha=0.1)
mesh_wm = Poly3DCollection(verts_wm[faces_wm], alpha=0.1)
mesh_csf = Poly3DCollection(verts_csf[faces_csf], alpha=0.1)

face_color = [0.5, 0.5, 1]
mesh_gm.set_facecolor(face_color)
mesh_wm.set_facecolor(face_color)
mesh_csf.set_facecolor(face_color)

ax.add_collection3d(mesh_gm)
ax.add_collection3d(mesh_wm)
ax.add_collection3d(mesh_csf)

ax.set_xlim(0, gm.shape[0])
ax.set_ylim(0, gm.shape[1])
ax.set_zlim(0, gm.shape[2])

plt.show()


# Save the isosurfaces as a DFS file
import nibabel as nib
from nibabel import freesurfer
import numpy as np

# Create a NIfTI image from the isosurfaces
data = np.zeros((256, 256, 256))
data[verts_gm[:, 0], verts_gm[:, 1], verts_gm[:, 2]] = 1
data[verts_wm[:, 0], verts_wm[:, 1], verts_wm[:, 2]] = 2
data[verts_csf[:, 0], verts_csf[:, 1], verts_csf[:, 2]] = 3

# Save the NIfTI image
img = nib.Nifti1Image(data, np.eye(4))
nib.save(img, "tissue_isosurfaces.nii.gz")

# Save the isosurfaces as a DFS file
freesurfer.write_geometry("tissue_isosurfaces.dfs", verts_gm, faces_gm, labels=np.ones(verts_gm.shape[0]))
freesurfer.write_geometry("tissue_isosurfaces.dfs", verts_wm, faces_wm, labels=np.ones(verts_wm.shape[0]))




# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load T1 image
t1_img = nib.load(input_path)

# Skull stripping
skullstrip = BET()
skullstrip.inputs.in_file = input_path
skullstrip.inputs.out_file = os.path.join(output_dir, "skullstripped.nii.gz")
skullstrip.run()

# Tissue classification
tissue_class = FAST()
tissue_class.inputs.in_files = os.path.join(output_dir, "skullstripped.nii.gz")
tissue_class.inputs.out_basename = "tissue_classified"
tissue_class.run()

# Output tissue probability maps
tissue_prob_files = [
    os.path.join(output_dir, "tissue_classified_pve_0.nii.gz"),  # GM
    os.path.join(output_dir, "tissue_classified_pve_1.nii.gz"),  # WM
    os.path.join(output_dir, "tissue_classified_pve_2.nii.gz"),  # CSF
]

# Saving tissue probability maps as separate NIfTI files
for i, tissue_prob_file in enumerate(tissue_prob_files):
    tissue_prob_map = nib.load(tissue_prob_file)
    nib.save(tissue_prob_map, os.path.join(output_dir, f"tissue_classified_{i}.nii.gz"))

print("Processing complete. Output files saved in:", output_dir)


