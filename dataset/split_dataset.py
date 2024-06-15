from sklearn.model_selection import KFold
from CT_pancreas_ids import PanCTDataset
from torch.utils.data import DataLoader, ConcatDataset

import os
import json

k_flods = 8
shuffle=True
root = '../../data/CT_Pancreas/Sloan_data'
depth_size = 32
num_samples = 12
is_transform = True
kfold = KFold(n_splits=k_flods, shuffle=shuffle)

train_pandataset = PanCTDataset(root=root,
                                depth_size=depth_size,
                                num_samples=num_samples,
                                is_transform=is_transform)

out_dict = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_pandataset)):
    print(train_ids)
    print('*'*20)
    print(test_ids)
    print('*'*20)
    out_dict_temp = {f'train_id fold_{fold}': train_ids.tolist(),
                     f'test_id fold_{fold}': test_ids.tolist(),}
    out_dict.update(out_dict_temp)

with open("split_dataset_8.json", "w") as f:
    json.dump(out_dict, f, indent=2)
