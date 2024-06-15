import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trans_3DUnet import get_model_dict
import monai
from torch.utils.data import DataLoader, Dataset
import glob
import nibabel as nib
import os
from synergynet import SynVNet_8h2s
# os.environ["CUDA_VISIBLE_DEVICES"]="7"




class pancreas(Dataset):
    def __init__(self, path):
        self.path = path
        self.x_paths = glob.glob(path+'/images/*.nii.gz')
    
    def __len__(self):
        return len(self.x_paths)
    
    def __getitem__(self, idx):

        image = nib.load(self.x_paths[idx]).get_fdata()
        image = torch.tensor(image).unsqueeze(0)
        image = monai.transforms.spatial.functional.resize(
            image, 
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

        image = (image - torch.mean(image)) / torch.std(image)

        img_name = self.x_paths[idx].split('/')[-1]

        label = nib.load(self.path+'/labels/'+img_name).get_fdata()
        label = torch.tensor(label).unsqueeze(0)
        label = monai.transforms.spatial.functional.resize(
            label, 
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

        return image, label


#PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE



def train_fn(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.
    for i, data in enumerate(loader):
        x = data[0].to(device)
        y = data[1].to(device)

        optimizer.zero_grad()
        y_hat,_ = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss/len(loader)


if __name__ == "__main__":

    device = torch.device('cuda:7')
    epochs = 250
    model_fn = get_model_dict("MaskTransUnet")
    model = model_fn(
                num_layers=[16, 64, 64, 128],
                roi_size_list=[40, 30, 20, 20],
                is_roi_list=[False, True, False, True],
                dim_input=1,
                dim_output=1,
                kernel_size=3
                )

    # model = SynVNet_8h2s()
    model.to(device)
    criterion = DiceBCELoss().to(device)

    train_ds = pancreas("/data/onkar/NeurIPS_Liver_unlabelled/training_t1")
    train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(epochs):

        loss = train_fn(model, optimizer, criterion, train_dataloader, device)

        print(f"Epoch {epoch + 1} loss: ", loss)
        if epoch % 10 == 0:
            checkpoint = {
                "model":model.state_dict()
            }
            torch.save(checkpoint, f"/data/onkar/NeurIPS_Liver_unlabelled/training_t1/neurips_models/{epoch+1}.pth")


