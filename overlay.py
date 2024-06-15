import numpy as np
import cv2
from PIL import Image

def get_overlay(image, colored_mask):
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.8, colored_mask, 0.6, 0)
    return overlay


class_dict_ACDC = {
        0:(0, 0, 0),
        1:(255, 0, 155),
        2:(255, 0, 155),
        3:(117, 200, 91),
}

def onehot_to_rgb(onehot, color_dict=class_dict_ACDC):
    onehot = np.int64(onehot)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[onehot==k] = color_dict[k]
    return np.uint8(output)

slice_no = 21
im = cv2.imread(f'/home/awd8324/onkar/TransUnet3D/img/{slice_no}.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

mask = cv2.imread("/home/awd8324/onkar/TransUnet3D/lab/17.png")
mask = mask.astype("uint8")[:,:,0]/255.
print(mask.shape)

mask = onehot_to_rgb(mask)

i2m = get_overlay(im, mask)

i2m = Image.fromarray(i2m)

file_name = "67"
mod = 't2s'
fold = "meddiff"

i2m.save(f"/home/awd8324/onkar/TransUnet3D/predections_Slices/{file_name}_{mod}_{fold}_{slice_no}.png")