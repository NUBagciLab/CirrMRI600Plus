import nibabel as nib
import glob 

files = glob.glob("/data/onkar/NeurIPS_Liver_unlabelled/training_t2/images/*.nii.gz")
print(len(files))
count = 0
# for f in files:
#     x = nib.load(f).get_fdata()
#     count += x.shape[-1]

# print(count)