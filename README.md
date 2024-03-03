## Skull Extraction Tool (SET)

**Description:**

This script leverages a deep learning model (UNet) to extract the skull from a given magnetic resonance imaging (MRI) image. 

**Requirements:**

* Python 3.x (https://www.python.org/downloads/)
* PyTorch (https://pytorch.org/)
* MONAI (https://monai.io/)
* Nibabel (https://nipy.org/nibabel/manual.html)
* NumPy (https://numpy.org/)
* Matplotlib (https://matplotlib.org/)

**Usage:**

```bash
python main.py -i <input_image> -m <model_file> -o <output_image> [--mask <mask_image>] [--device cpu|cuda]
```

**Arguments:**

* -i, --input: Path to the input MRI image file. (Required)
* -m, --model: Path to the trained UNet model file (.pth). (Required)
* -o, --output: Path to save the extracted skull image. (Required)
* --mask: (Optional) Path to save the predicted skull mask image.
* --device: Device to use for computation, either "cpu" or "cuda" (defaults to "cuda" if available).

**Example:**

```bash
python main.py -i input.nii.gz -m model.pth -o skull.nii.gz --mask mask.nii.gz --device cpu
```


** Notes: **

This script assumes the input data is in NIfTI format.
The model expects the input image to be preprocessed (e.g., scaled, resized) before feeding it to the network.
Ensure the model architecture and training configuration are compatible with the provided script.

** Disclaimer: **

This script is for educational purposes only and may not be suitable for clinical applications.
