import nibabel as nib
import monai
import matplotlib.pyplot as plt
import torch

image = nib.load("/data/onkar/NeurIPS_Liver_unlabelled/training_t2/images/67.nii.gz").get_fdata()
label = nib.load("/data/onkar/NeurIPS_Liver_unlabelled/training_t2/labels/67.nii.gz").get_fdata()

print(image.shape, label.shape)

image = torch.tensor(image).unsqueeze(0)

image = monai.transforms.spatial.functional.resize(
            image, 
            out_size=(320, 320, 32), 
            mode="nearest", 
            align_corners=None, 
            dtype=None, 
            input_ndim=3, 
            anti_aliasing=False, 
            anti_aliasing_sigma=None, 
            lazy=False, 
            transform_info=None
        ).squeeze(0).numpy()

label = torch.tensor(label).unsqueeze(0)

label = monai.transforms.spatial.functional.resize(
            label, 
            out_size=(320, 320, 32), 
            mode="nearest", 
            align_corners=None, 
            dtype=None, 
            input_ndim=3, 
            anti_aliasing=False, 
            anti_aliasing_sigma=None, 
            lazy=False, 
            transform_info=None
        ).squeeze(0).numpy()



for i in range(image.shape[-1]):
    plt.imsave(f"/home/awd8324/onkar/TransUnet3D/img/{i}.png", image[:,:,i], cmap='gray')
    plt.imsave(f"/home/awd8324/onkar/TransUnet3D/lab/{i}.png", label[:,:,i], cmap='gray')



