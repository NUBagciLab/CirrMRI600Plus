import os
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset, Subset
from torch.cuda.amp import autocast
import numpy as np

from dataset.CT_pancreas_multi_class import EvaPanCTDataset
from model.trans_3DUnet import get_model_dict
from loss.multi_criterions import get_criterions

import monai
from monai.inferers import sliding_window_inference

def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_data', type=str,
                        default='../../data/CT_Pancreas/Sloan_data',
                        help='direction for the dataset')
    # 12 -> 20220130-15_2
    parser.add_argument('--pretrained_dir', type=str,
                        default='./out/log/20220201-23_2', help='pretrained dir')
    parser.add_argument('--model_name', type=str,
                        default='MaskTransUnet', help='model name for training')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='patient batch size')
    parser.add_argument('--depth_size', type=int,
                        default=32, help='patient depth size')
    # [16, 32, 64, 128, 256]
    # [32, 64, 64, 128, 256]
    parser.add_argument('--num_layers', type=list, 
                        default=[16, 32, 64, 128, 256], help='number of layer for each layer')
    # reference 320 160 80 40 20 [128, 80, 50, 30, 32]
    # 256-128-64-32-16 65
    parser.add_argument('--roi_size_list', type=list, 
                        default=[100, 65, 40, 25, 10], help='size of roi for each layer')
    # False, True, True, True, True
    parser.add_argument('--is_roi_list', type=list, 
                        default=[False, True, True, True, True], help='using roi for each layer')
    '''
    parser.add_argument('--num_layers', type=list, 
                        default=[16, 32, 32, 64], help='number of layer for each layer')
    '''
    parser.add_argument('--dim_input', type=int,
                        default=1, help='input dimension or modality')
    parser.add_argument('--dim_output', type=int,
                        default=3, help='output dimension or classes')
    parser.add_argument('--kernel_size', type=int,
                        default=3, help='kernel_size for convolution')

    parser.add_argument('--device', type=str,
                        default='cuda', help='device for training')
    parser.add_argument('--criterion_list', type=list,
                        default=['DiceClassLoss0', 'DiceClassLoss', 'DiceClassLoss2', 'Recall', 'Precision', 'Recall2', 'Precision2','LocalizationLoss'],
                        help='criterion')

    parser.add_argument('--is_save', type=bool,
                        default=False, help='save prediction or not')
    parser.add_argument('--saved_folder', type=str,
                        default='./prediction/test', 
                        help='saved folder dir')

    args = parser.parse_args()
    return args

def get_model(args, fold_num, device):
    model_fn = get_model_dict(args.model_name)
    model = model_fn(num_layers=args.num_layers,
                     roi_size_list=args.roi_size_list,
                     is_roi_list=args.is_roi_list,
                     dim_input=args.dim_input,
                     dim_output=args.dim_output,
                     kernel_size=args.kernel_size)
    
    pretrain_dir = os.path.join(args.pretrained_dir, f'fold_{fold_num}', 'temp_model.pt')
    # state_dict = torch.load(pretrain_dir).state_dict()
    state_dict = torch.load(pretrain_dir)

    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model.to(device))
    return model

def main(args):
    fold_nums = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    num_device = torch.cuda.device_count()
    root = args.dir_data
    depth_size = args.depth_size
    sw_batch_size = 4

    with open('split_dataset_8.json', 'r') as f:
        dataset_ids = json.load(f)

    criterions = get_criterions(args.criterion_list)
    final_loss_list = [0] * len(criterions)
    roi_size = 512
    center = 256
    name_list = sorted(os.listdir(os.path.join(root, 'image')))
    # up_sample = torch.nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
    post_processing = monai.transforms.KeepLargestConnectedComponent(applied_labels=[1, 2], independent=False, connectivity=3)

    for fold_num in range(fold_nums):
        test_ids = dataset_ids[f'test_id fold_{fold_num}']
        eval_pandataset = EvaPanCTDataset(root=root,
                                           depth_size=depth_size,
                                           ids=test_ids[:-1])
        eval_panDl = DataLoader(dataset=eval_pandataset, batch_size=args.batch_size,
                                num_workers=12, shuffle=False)

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = get_model(args, fold_num, device=device)
        
        model.eval()
        summary_patient_loss = []
        total_loss_list = [0] * len(criterions)
        threshold = 0.5
        num_classes = args.dim_output

        if not os.path.exists(args.saved_folder):
            os.makedirs(args.saved_folder)

        for i, (images, masks) in enumerate(eval_panDl):
            name = name_list[test_ids[i]]
            print(name)

            images, masks = images.to(device), masks.to(device).long()
            with torch.no_grad():
                n, c, h, w, d = masks.shape
                label = masks.flatten(2).transpose(1, 2).squeeze(2)
                # print(torch.max(label))
                label = F.one_hot(label , num_classes = num_classes)
                label = label.transpose_(1, 2)
                label = torch.reshape(label, (n, num_classes, h, w, d))
            # predict = torch.zeros((masks.size(0), 2, masks.size(2), masks.size(3), masks.size(4)), device=masks.device)
            patient_loss_list = [0] * len(criterions)

            with torch.no_grad():
                with autocast():
                    predict = sliding_window_inference(images, (roi_size, roi_size, depth_size), sw_batch_size, model, overlap=0.6, sigma_scale=0)
                    # print(predict_center.shape)
                    '''
                    predict2 = predict
                    '''
                    predict2 = torch.round(predict)
                    predict2 = predict2.float().squeeze(0)
                    predict2 = post_processing(predict2)
                    predict2 = predict2.unsqueeze(0)
                    predict2[:, 0] = 1 - predict2[:, 1] - predict2[:, 2]
                    loss_list = [l(predict2, label).item() for l in criterions.values()]
                    
            if args.is_save:
                # predict = up_sample(predict)
                with torch.no_grad():
                    temp_out = torch.argmax(predict2, dim=1)
                print(temp_out.shape)
                # temp_out = predict
                temp_out = temp_out.squeeze_().permute((2, 0, 1)).cpu().numpy()

                np.save(os.path.join(args.saved_folder, '{:0>4}'.format(name)+'_multi'), temp_out)
            
            for loss_name, loss_value in zip(criterions.keys(), loss_list):
                print(f'eval patient average {loss_name}', loss_value)

            for index, loss_value in enumerate(patient_loss_list):
                patient_loss_list[index] = loss_list[index]
                total_loss_list[index] += patient_loss_list[index]
            summary_patient_loss.append(patient_loss_list)

        for index, loss_value in enumerate(total_loss_list):
            total_loss_list[index] = loss_value / (i+1)
            final_loss_list[index] += total_loss_list[index]

        for loss_name, loss_value in zip(criterions.keys(), total_loss_list):
            print(f'eval total average {loss_name} loss', loss_value)

        out_dict = {f'patient_{fold_num}': summary_patient_loss,
                    f'summary_{fold_num}': total_loss_list}

    for index, loss_value in enumerate(final_loss_list):
        final_loss_list[index] = loss_value / (fold_num+1)

    for loss_name, loss_value in zip(criterions.keys(), final_loss_list):
        print(f'eval final average {loss_name} loss', loss_value)

    with open('summary_4_fold.json', 'w') as f:
            json.dump(out_dict, f, indent=4)


if __name__ =='__main__':
    args = get_parse()
    main(args)
