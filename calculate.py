import numpy as np
import nibabel as nib


lab = nib.load("/data/onkar/NeurIPS_Liver_unlabelled/training_t1/labels/304.nii.gz")
# print(lab.shape)
lab_data = lab.get_fdata()
lab_data = np.concatenate([lab_data[:,:,10:], lab_data[:,:,:10]], axis=-1) 
lab_data = lab_data.astype("uint8")
print(lab_data.shape)

lab_img = nib.Nifti1Image(lab_data, lab.affine, lab.header)
nib.save(lab_img, "304_t1_synet3d.nii.gz")