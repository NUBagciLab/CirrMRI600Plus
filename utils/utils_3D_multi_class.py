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
                   criterions, criterion_weight, device, writer, 
                   patient_epochs, patient_batchsize, global_step,
                   dynamic_weight, num_classes=3):
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
        # print('start epoch')
        # print(images.shape)
        # print(masks.shape)
        images, masks = torch.flatten(images, start_dim=0, end_dim=1),\
                            torch.flatten(masks, start_dim=0, end_dim=1)

        frames = images.size(0)
        patient_level_loss = 0
        patient_loss_list = [[0 for _ in range(len(criterions[i]))]
                                for i in range(len(criterions))]
        optimizer.zero_grad()

        for j in range(patient_epochs):
            # print('start batch')
            frame_index = torch.randint(low=0, high=frames, size=(patient_batchsize,))

            # select the corresponding frame and unsqueeze the input dimension level
            batch_images, batch_masks = images[frame_index], masks[frame_index]
            batch_images, batch_masks = batch_images.to(device, non_blocking=True), batch_masks.to(device, non_blocking=True)
            # batch_masks = batch_masks.long()

            with torch.no_grad():
                n, c, h, w, d = batch_masks.shape
                # print('org', batch_masks.shape)
                batch_label = batch_masks.flatten(2).transpose(1, 2).squeeze(2)
                # print('step1', batch_label.shape)
                batch_label = F.one_hot(batch_label , num_classes = num_classes)
                # print('step2', batch_label.shape)
                batch_label = batch_label.transpose_(1, 2)
                # print('step3', batch_label.shape)
                batch_label = torch.reshape(batch_label, (n, num_classes, h, w, d))
            loss_list = []
            # temp_masks = batch_masks
            with autocast():
                temp_masks = F.max_pool3d(batch_masks.float(), kernel_size=(2, 2, 1), stride=(2, 2, 1))
                predict, roi_mask = model(batch_images)
                for indice_out in range(len(dynamic_weight)):
                    if indice_out == 0:
                        temp_loss = [criterions_w*l(predict, batch_label) for l, criterions_w in zip(criterions[-indice_out-1].values(), criterion_weight)]

                    else:
                        with torch.no_grad():
                            n, c, h, w, d = temp_masks.shape
                            temp_label = temp_masks.flatten(2).transpose(1, 2).squeeze(2)
                            temp_label = temp_label.to(torch.long)
                            # print(torch.max(temp_label))
                            temp_label = F.one_hot(temp_label , num_classes = num_classes)
                            temp_label = temp_label.transpose_(1, 2)
                            temp_label = torch.reshape(temp_label, (n, num_classes, h, w, d))
            
                        temp_loss = [criterions_w*l(roi_mask[-indice_out], temp_label) for l, criterions_w in zip(criterions[-indice_out-1].values(), criterion_weight)]

                        with torch.no_grad():
                            if indice_out % 2== 0:
                                temp_masks = F.max_pool3d(temp_masks, kernel_size=2, stride=2)
                            else:
                                temp_masks = F.max_pool3d(temp_masks, kernel_size=(2, 2, 1), stride=(2, 2, 1))

                    for index, loss_value in enumerate(temp_loss):
                        # print(f'error output {indice_out} and index is {index}, name of {criterion_name_total[-indice_out-1][index]}')
                        # print(f'loss is {loss_value.item()}')
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
            # print('end--')
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
                  patient_epochs, patient_batchsize,
                  global_step, num_classes=3):
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
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            n, c, h, w, d = masks.shape
            label = masks.flatten(2).transpose(1, 2).squeeze(2)
            # print(torch.max(label))
            label = F.one_hot(label , num_classes = num_classes)
            label = label.transpose_(1, 2)
            label = torch.reshape(label, (n, num_classes, h, w, d))
        
        with torch.no_grad():
            with autocast():
                predict = sliding_window_inference(images, (roi_size, roi_size, depth_size), sw_batch_size, model, overlap=0.6, sigma_scale=0)
                # print(predict_center.shape)

                loss_list = [l(predict, label).item() for l in criterions.values()]

        print(f'eval: patient average loss', sum(loss_list))
        for index, loss_value in enumerate(loss_list):
            '''
            if torch.any(torch.isnan(loss_value)):
                print(f'error output {index}')
                print(f'loss is {loss_value}')
            '''
            patient_total_list[index] += loss_value
    batch_level_loss = sum(patient_total_list) / (i+1)

    writer.add_scalar('eval/total_loss', batch_level_loss, global_step=global_step)
    for loss_name, loss_value in zip(criterions.keys(), patient_total_list):
        if loss_name == 'DiceClassLoss':
            out_loss = loss_value / (i+1)
        if loss_name == 'DiceClassLoss2':
            out_loss = out_loss + loss_value / (i+1)
        writer.add_scalar(f'eval/{loss_name}', loss_value / (i+1), global_step=global_step)

    print(f'eval: batch average loss', batch_level_loss)
    return out_loss, global_step+1

def save_model(model, model_dir:str):
    r'''
    save the model if the performance is better
    '''
    torch.save(model, model_dir)
