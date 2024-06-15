import torch
import monai

def get_predection(x, y, model):
    model.eval()
    y = torch.tensor(y).unsqueeze(0)

    y = monai.transforms.spatial.functional.resize(
            y, 
            out_size=(256, 256, 80), 
            mode="nearest", 
            align_corners=None, 
            dtype=None, 
            input_ndim=3, 
            anti_aliasing=False, 
            anti_aliasing_sigma=None, 
            lazy=False, 
            transform_info=None
        ).squeeze(0).numpy()
    
    x = monai.transforms.spatial.functional.resize(
            x, 
            out_size=(256, 256, 80), 
            mode="nearest", 
            align_corners=None, 
            dtype=None, 
            input_ndim=3, 
            anti_aliasing=False, 
            anti_aliasing_sigma=None, 
            lazy=False, 
            transform_info=None
        )

    with torch.no_grad():
        pred = model(x)
    pred = torch.sigmoid(pred).squeeze(0).numpy()

    return pred, y
    