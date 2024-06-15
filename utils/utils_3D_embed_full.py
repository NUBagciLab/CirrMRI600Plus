'''
Epoch utils function for training and evluation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import monai
from monai.inferers import sliding_window_inference

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def get_weight(t, T, default_weight:float=0.2, initial_weight:float=1, final_weight:float=1.0):
    t = max(t, 0)
    weight = min(initial_weight + default_weight*np.exp(t/(5*T)), final_weight)
    return weight

def train_on_epoch(model, dataloader, optimizer, step_times,
                   criterions, device, writer, 
                   patient_epochs, patient_batchsize, global_step,
                   dynamic_weight):
    r'''
    Train the model on each epoch
    Args:
        model: the model for training
        dataloader: the dataloader for training, returning should be image-mask pair
        optimizer: the optimizer for training
        criterions: the defination for loss function, should be dict
        criterion_weight: the weight for each corresponding loss function
        device: cuda if gpu is available
        writer: the tensorboard write
        patient_epochs: the patient level epochs
        patient_batchsize: the batch size for each patient
        global_step: global step for training
        dynamic_weight: the dynamic weight list
        surface_distance: the surface distance list
    '''
    model.train()
    batch_level_loss = 0

    criterion_name_total = [list(criterions[i].keys()) for i in range(len(criterions))]
    for i, (images, masks) in enumerate(dataloader):
        images, masks = torch.flatten(images, start_dim=0, end_dim=1),\
                            torch.flatten(masks, start_dim=0, end_dim=1)

        frames = images.size(0)
        patient_level_loss = 0
        patient_loss_list = [[0 for _ in range(len(criterions[i]))]
                                for i in range(len(criterions))]
        optimizer.zero_grad()

        for j in range(patient_epochs):
            frame_index = torch.randint(low=0, high=frames, size=(patient_batchsize,))

            # select the corresponding frame and unsqueeze the input dimension level
            batch_images, batch_masks = images[frame_index], masks[frame_index]
            batch_images, batch_masks = batch_images.to(device, non_blocking=True), batch_masks.to(device, non_blocking=True)
            loss_list = []
            # temp_masks = batch_masks
            with autocast():
                temp_masks = F.max_pool3d(batch_masks.float(), kernel_size=(2, 2, 1), stride=(2, 2, 1))
                predict, roi_mask = model(batch_images)
                for indice_out in range(len(dynamic_weight)):
                    if indice_out == 0:
                        temp_loss = [l(predict, batch_masks.long()) for l in criterions[-indice_out-1].values()]
                    else:
                        temp_loss = [l(roi_mask[-indice_out], temp_masks.long()) for l in criterions[-indice_out-1].values()]

                        with torch.no_grad():
                            if indice_out % 2== 0:
                                temp_masks = F.max_pool3d(temp_masks, kernel_size=2, stride=2)
                            else:
                                temp_masks = F.max_pool3d(temp_masks, kernel_size=(2, 2, 1), stride=(2, 2, 1))

                    for index, loss_value in enumerate(temp_loss):
                        patient_loss_list[-indice_out-1][index] += loss_value.item()
                    loss_list.append(temp_loss)

                total_loss = sum([sum(loss)*weight for loss, weight in zip(loss_list, dynamic_weight)])
            
            patient_level_loss +=  total_loss.item()
            total_loss = total_loss / step_times
            scaler.scale(total_loss).backward()

            if (j+1) % step_times == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        patient_level_loss = patient_level_loss / (j+1)
        '''
        print(loss_list)
        print(patient_loss_list)
        print(temp_loss)
        print(criterion_name_total[indice_out])
        print(patient_loss_list[indice_out])
        '''
        writer.add_scalar('train/total_loss', patient_level_loss, global_step=(global_step + i))
        for indice_out in range(len(dynamic_weight)):
            for loss_name, loss_value in zip(criterion_name_total[indice_out],
                                             patient_loss_list[indice_out]):
                writer.add_scalar(f'train/{loss_name}'+' layer'+f'{str(indice_out)}', 
                                  loss_value / (j+1), global_step=(global_step + i))

        print(f'train: patient average loss', patient_level_loss)
        batch_level_loss += patient_level_loss

    batch_level_loss = batch_level_loss / (i + 1)
    global_step += i
    writer.add_scalar('lr_rate', optimizer.param_groups[0]['lr'], global_step=global_step)

    print(f'train: batch average loss', batch_level_loss)
    return batch_level_loss, global_step

def eval_on_epoch(model, dataloader, 
                  criterions, device, writer, 
                  patient_epochs, patient_batchsize, global_step):
    r'''
    eval the model on each epoch
    Args:
        model: the model for training
        dataloader: the dataloader for eval, returning should be image-mask pair
        criterions: the defination for loss function, should be dict
        criterion_weight: the weight for each corresponding loss function
        device: cuda if gpu is available
        writer: the tensorboard write
        patient_epochs: the patient level epochs
        patient_batchsize: the batch size for each patient
        global_step: global step for training
    '''
    model.eval()
    roi_size = 512
    depth_size = 64
    sw_batch_size = 2*patient_batchsize

    threshold = 0.5
    out_loss = 0
    patient_total_list = [0] * len(criterions)

    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device).long()

        with torch.no_grad():
            with autocast():
                predict = sliding_window_inference(images, (roi_size, roi_size, depth_size), sw_batch_size, model, overlap=0.6, sigma_scale=0)
                # print(predict_center.shape)
                '''
                predict2 = predict
                '''
                predict2 = (predict >= threshold).float()

                loss_list = [l(predict2, masks.long()).item() for l in criterions.values()]

        print(f'eval: patient average loss', sum(loss_list))
        for index, loss_value in enumerate(loss_list):
            patient_total_list[index] += loss_value
    batch_level_loss = sum(patient_total_list) / (i+1)

    writer.add_scalar('eval/total_loss', batch_level_loss, global_step=global_step)
    for loss_name, loss_value in zip(criterions.keys(), patient_total_list):
        if loss_name == 'DiceClassLoss':
            out_loss = loss_value / (i+1)
        writer.add_scalar(f'eval/{loss_name}', loss_value / (i+1), global_step=global_step)

    print(f'eval: batch average loss', batch_level_loss)
    return out_loss, global_step+1

def save_model(model, model_dir:str):
    r'''
    save the model if the performance is better
    '''
    torch.save(model, model_dir)
