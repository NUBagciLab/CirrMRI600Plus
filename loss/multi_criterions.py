import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class DiceLoss(nn.Module):
    '''
    General Dice coefficient loss
    '''
    def __init__(self, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        cross_region = 2*torch.sum(predict_reshape * target_onehot, dim=1) + self.eps
        sum_region = torch.sum(predict_reshape + target_onehot, dim=1) + self.eps
        dice = torch.mean(cross_region / sum_region)
        return 1 - dice

class DiceClassLoss0(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =0, eps:float = 1e-9):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = 1 - class_reshape[:, :, 0]
        class_target = 1 - target_onehot[:, :, 0]

        cross_region = 2*torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict + class_target, dim=-1) + self.eps
        class_dice = torch.mean(cross_region / sum_region)
        return 1- class_dice


class DiceClassLoss(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-9):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = 2*torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict + class_target, dim=-1) + self.eps
        class_dice = torch.mean(cross_region / sum_region)
        return 1- class_dice

class DiceClassLoss2(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =2, eps:float = 1e-9):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = 2*torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict + class_target, dim=-1) + self.eps
        class_dice = torch.mean(cross_region / sum_region)
        return 1- class_dice


class RegionDiceClassLoss(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor, 
                dist:Tensor=None, surface_distance=None):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
            dist: distance maxtrix to surface, size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        assert self.class_index < n_channel, 'index beyond output classs'
        predict_reshape = predict.flatten(2).transpose(2, 1)
        class_predict = predict_reshape[:, :, self.class_index]
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        '''
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        class_target = target_onehot[:, :, self.class_index]
        '''
        class_target = target_reshape
        dist_weight = 0.5
        '''
        if dist is not None:
            with torch.no_grad():
                if surface_distance != 0:
                    dist_weight = torch.sigmoid(dist/surface_distance)
                    dist_weight = dist_weight.flatten(2).transpose(2, 1).squeeze(2)

        dist_weight = 0.5
        '''
        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(dist_weight*class_predict + (1-dist_weight)*class_target, dim=-1) + self.eps

        class_dice = torch.mean(cross_region / sum_region)
        return 1- class_dice


class DistributionLoss(nn.Module):
    '''
    Distribution loss for three distribution
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-7):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
        self.mask_threshold = 0.5
        self.mask_region = 0.05

    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        predict = torch.sigmoid((predict-self.mask_threshold)/self.mask_region)
        assert self.class_index < n_channel, 'index beyond output classs'
        predict = predict[:, self.class_index, :].unsqueeze_(1)

        n_dim = predict.dim() - 2
        for i in range(n_dim):
            if i!= 0:
                dim_label = target.flatten(3)
                dim_label = target.flatten(3)
            else:
                dim_predict = predict.transpose(2, (i+2))
                dim_label = target.transpose(2, (i+2))

                dim_predict = dim_predict.flatten(3)
                dim_label = dim_label.flatten(3)

            dim_predict = torch.sum(dim_predict, dim=-1)
            dim_label = torch.sum(dim_label, dim=-1)
            if i== 0:
                dim_loss = self.dis_loss(dim_predict, dim_label, eps=self.eps)
            else:
                dim_loss += self.dis_loss(dim_predict, dim_label, eps=self.eps)

        dim_loss = dim_loss / n_dim
        return dim_loss 

    @staticmethod
    def dis_loss(predict:Tensor, target: Tensor, eps:float):
        '''
        predict: [N, 1, H]
        target: [N, 1, H]
        '''
        dist_pred = torch.cumsum(predict, dim=-1) / (torch.sum(predict, dim=-1, keepdim=True) + eps)
        dist_target = torch.cumsum(target, dim=-1) / (torch.sum(target, dim=-1, keepdim=True) + eps)
        # dist_weight = (2*dist_target-1)**2
        dist_weight = 1
        dim_loss = torch.mean(dist_weight * torch.abs(dist_pred-dist_target))
        return dim_loss


class LocalizationLoss(nn.Module):
    '''
    Distribution loss for three distribution
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-6):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
        self.mask_threshold = 10
        self.mask_region = 0.05

    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
            dist: distance maxtrix to surface, size [N, 1, H, W, D...]
        '''
        predict_r =  (1- predict[:, 0]).clone().unsqueeze_(1)
        target_r = (1 - target[:, 0]).clone().unsqueeze_(1)
        # print(predict.shape)

        n_dim = predict_r.dim() - 2
        for i in range(n_dim):
            if i!= 0:
                dim_predict = predict_r.flatten(3)
                dim_label = target_r.flatten(3)
            else:
                dim_predict = predict_r.transpose(2, (i+2))
                dim_label = target_r.transpose(2, (i+2))

                dim_predict = dim_predict.flatten(3)
                dim_label = dim_label.flatten(3)

            # print('dim predict', dim_predict.shape)
            dim_predict = torch.sum(dim_predict, dim=-1)
            dim_predict = torch.sigmoid(dim_predict-self.mask_threshold)
            
            dim_label = torch.sum(dim_label, dim=-1)
            dim_label = torch.sigmoid(dim_label-self.mask_threshold)

            if i== 0:
                dim_loss = self.dis_loss(dim_predict, dim_label, eps=self.eps)
            else:
                dim_loss += self.dis_loss(dim_predict, dim_label, eps=self.eps)

        dim_loss = dim_loss / n_dim
        return dim_loss 

    @staticmethod
    def dis_loss(predict:Tensor, target: Tensor, eps:float):
        '''
        predict: [N, 1, H]
        target: [N, 1, H]
        '''
        dist_pred = torch.cumsum(predict, dim=-1) / (torch.sum(predict, dim=-1, keepdim=True) + eps)
        dist_target = torch.cumsum(target, dim=-1) / (torch.sum(target, dim=-1, keepdim=True) + eps)
        # dist_weight = (2*dist_target-1)**2
        # dist_weight = 1
        dim_loss = torch.mean(torch.abs(dist_pred-dist_target))
        return dim_loss


class MaskLoss(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, 1, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        # calculate the foreground 
        class_predict = predict_reshape.squeeze(2)
        class_target = target.flatten(2).transpose(2, 1).squeeze(2)
        '''
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        class_target = target_onehot[:, :, self.class_index]
        class_target = target_reshape
        '''
        cross_region = 2*torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict + class_target, dim=-1) + self.eps
        '''
        if torch.any(torch.isnan(class_predict)):
            print('final predict')
            print(torch.min(class_predict))
        '''
        class_dice = torch.mean(cross_region / sum_region)
        return 1- class_dice


class Recall(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_target, dim=-1) + self.eps

        recall = torch.mean(cross_region / sum_region)
        return recall

class Recall2(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =2, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_target, dim=-1) + self.eps

        recall = torch.mean(cross_region / sum_region)
        return recall


class RecallLoss(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_target, dim=-1) + self.eps

        recall = torch.mean(cross_region / sum_region)
        return 1-recall


class Precision(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict, dim=-1) + self.eps

        precision = torch.mean(cross_region / sum_region)
        return precision


class Precision2(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =2, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict, dim=-1) + self.eps

        precision = torch.mean(cross_region / sum_region)
        return precision


class PrecisionLoss(nn.Module):
    '''
    Dice coefficient loss for certain class
    Args:
        class_index: the index for certain class
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.class_index = class_index
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        class_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        class_predict = class_reshape[:, :, self.class_index]
        class_target = target_onehot[:, :, self.class_index]

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict, dim=-1) + self.eps

        precision = torch.mean(cross_region / sum_region)
        return 1 - precision


class BalanceDiceLoss(nn.Module):
    '''
    General Dice coefficient loss
    '''
    def __init__(self, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        class_weight = 1 / (torch.sum(target_onehot, dim=1, keepdim=True) + self.eps)**2
        cross_region = 2*torch.sum(predict_reshape * target_onehot * class_weight, dim=(1, 2)) + self.eps
        sum_region = torch.sum((predict_reshape + target_onehot) * class_weight, dim=(1, 2)) + self.eps

        dice = torch.mean(cross_region / sum_region)
        return 1 - dice

class BalanceDiceLoss2(nn.Module):
    '''
    General Dice coefficient loss
    '''
    def __init__(self, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict = predict.flatten(2).transpose(2, 1)
        predict_reshape = predict[:, :, 1:]
        target = target.flatten(2).transpose(2, 1)
        target_onehot = target[:, :, 1:]

        class_weight = 1 / (torch.sum(target_onehot, dim=1, keepdim=True) + self.eps)**2
        cross_region = 2*torch.sum(predict_reshape * target_onehot * class_weight, dim=(1, 2)) + self.eps
        sum_region = torch.sum((predict_reshape + target_onehot) * class_weight, dim=(1, 2)) + self.eps

        dice = torch.mean(cross_region / sum_region)
        return 1 - dice


class IOULoss(nn.Module):
    '''
    IOU coefficient loss
    '''
    def __init__(self, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        cross_region = torch.sum(predict_reshape * target_onehot, dim=1) + self.eps
        sum_region = torch.sum(predict_reshape + target_onehot, dim=1) + self.eps
        iou = torch.mean(cross_region / (sum_region - cross_region))
        return 1 - iou



class FocalLoss(nn.Module):
    '''
    General focal loss
    Args:
        gamma: the value for balancing difficult samples
    '''
    def __init__(self, gamma:float = 2,eps:float = 1e-9):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)
        
        cross_value = - (1 - predict_reshape) ** self.gamma * \
                        target_onehot * torch.log(predict_reshape)
        focal_loss = torch.mean(cross_value)
        return focal_loss


class CrossEntroLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        log_predict_reshape = torch.log(torch.clamp(predict_reshape, min=1e-6))
        weight = torch.sum(predict_reshape, dim=1, keepdim=True) + self.eps
        total_sum = torch.sum(target_onehot, dim=(1, 2), keepdim=True)
        weight = (total_sum - weight) / total_sum
        cross_value = - weight*(1-predict_reshape)*target_onehot * log_predict_reshape
        cross_value = torch.mean(cross_value)

        return cross_value

class ClassifyLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        ndim = predict_reshape.shape[2]
        weight_list = torch.arange(end=ndim, device=predict.device).unsqueeze(0).unsqueeze(0)
        target_class = torch.sum(weight_list * target_onehot, dim=-1)
        predict_reshape = torch.sum(weight_list * predict_reshape, dim=-1)
        loss = torch.sum((1-target_onehot[:,:,0])*(predict_reshape-target_class)**2) / (torch.sum(1-target_onehot[:,:,0]) + self.eps)
        return loss



class CrossEntroLoss0(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict = predict.flatten(2).transpose(2, 1)
        predict_reshape = torch.stack([predict[:,:,0], 1-predict[:,:,0]], dim=-1)
        target = target.flatten(2).transpose(2, 1)
        target_onehot = torch.stack([target[:,:,0], 1-target[:,:,0]], dim=-1)

        log_predict_reshape = torch.log(torch.clamp(predict_reshape, min=1e-6))
        weight = torch.sum(predict_reshape, dim=1, keepdim=True) + self.eps
        total_sum = torch.sum(target_onehot, dim=(1, 2), keepdim=True)
        weight = (total_sum - weight) / total_sum
        cross_value = - weight*(1-predict_reshape)*target_onehot * log_predict_reshape
        cross_value = torch.mean(cross_value)

        return cross_value


class MSEcLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_onehot = target.flatten(2).transpose(2, 1)

        return super().forward(predict_reshape.float(), target_onehot.float())

Loss_Dict = {
    'DiceLoss': DiceLoss,
    'DiceClassLoss0': DiceClassLoss0,
    'DiceClassLoss': DiceClassLoss,
    'DiceClassLoss2': DiceClassLoss2,
    'BalanceDiceLoss': BalanceDiceLoss,
    'BalanceDiceLoss2':BalanceDiceLoss2,
    'IOULoss': IOULoss,
    'FocalLoss': FocalLoss,
    'CrossEntroLoss': CrossEntroLoss,
    'CrossEntroLoss0': CrossEntroLoss0,
    'MSELoss': MSEcLoss,
    'Recall':Recall,
    'Precision': Precision,
    'Recall2':Recall2,
    'Precision2': Precision2,
    'RecallLoss':RecallLoss,
    'PrecisionLoss': PrecisionLoss,
    'DistributionLoss': DistributionLoss,
    'LocalizationLoss': LocalizationLoss,
    'ClassifyLoss': ClassifyLoss
}

def get_criterions(name_list: list):
    '''
    Return the loss dict from name list
    Args:
        name_list: name list
    '''
    loss_dict = {}
    for name in name_list:
        loss_dict[name] = Loss_Dict[name]()
    return loss_dict