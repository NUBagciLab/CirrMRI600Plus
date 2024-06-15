from torch.utils.data import DataLoader, Dataset
import torch
from model.trans_3DUnet import get_model_dict
from train import pancreas
from medcam import medcam
from synergynet import SynVNet_8h2s
import monai

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model_fn = get_model_dict("MaskTransUnet")
# model = model_fn(
#             num_layers=[16, 64, 64, 128],
#             roi_size_list=[40, 30, 20, 20],
#             is_roi_list=[False, True, False, True],
#             dim_input=1,
#             dim_output=1,
#             kernel_size=3
#             )

model = monai.networks.nets.SwinUNETR(
img_size=(256, 256, 64), 
in_channels=1,
out_channels=1
)


train_ds = pancreas("/data/onkar/NeurIPS_Liver_unlabelled/training_t1")
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

# net = torch.load("/data/onkar/NeurIPS_Liver_unlabelled/training_t1/neurips_models/1.pth")
# model.load_state_dict(net['model'])
model.to(device)
print(model)

model = medcam.inject(model, output_dir="attention_maps", save_maps=True, backend="gcam")
model.eval()

for i, data in enumerate(train_dataloader):
    x = data[0].to(device)
    out = model(x)
    # out = out.detach().cpu().numpy()[0,0]
    print(out.shape)
    break









