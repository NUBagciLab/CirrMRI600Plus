from cProfile import label
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
            target: size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        # target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        # temp solution for saving memory
        target_onehot = torch.stack([1-target_reshape, target_reshape], dim=-1)

        cross_region = 2*torch.sum(predict_reshape * target_onehot, dim=1) + self.eps
        sum_region = torch.sum(predict_reshape + target_onehot, dim=1) + self.eps
        dice = torch.mean(cross_region / sum_region)
        return 1 - dice


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
            target: size [N, 1, H, W, D...]
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
        cross_region = 2*torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region = torch.sum(class_predict + class_target, dim=-1) + self.eps
        '''
        if torch.any(torch.isnan(class_predict)):
            print('final predict')
            print(torch.min(class_predict))
        '''
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
        n_channel = predict.size(1)
        predict_r = predict[:, self.class_index].clone().unsqueeze_(1)
        # print(predict.shape)

        n_dim = predict_r.dim() - 2
        for i in range(n_dim):
            if i!= 0:
                dim_predict = predict_r.flatten(3)
                dim_label = target.flatten(3)
            else:
                dim_predict = predict_r.transpose(2, (i+2))
                dim_label = target.transpose(2, (i+2))

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
        dim_loss = 8*torch.mean(torch.abs(dist_pred-dist_target))
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
            target: size [N, 1, H, W, D...]
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
            target: size [N, 1, H, W, D...]
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
            target: size [N, 1, H, W, D...]
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
            target: size [N, 1, H, W, D...]
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
            target: size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        # target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        target_onehot = torch.stack([1-target_reshape, target_reshape], dim=-1)

        class_weight = 1 / (torch.sum(target_onehot, dim=1, keepdim=True) + self.eps)**2

        cross_region = 2*torch.sum(predict_reshape * target_onehot * class_weight, dim=(1, 2)) + self.eps
        sum_region = torch.sum((predict_reshape + target_onehot) * class_weight, dim=(1, 2)) + self.eps

        dice = torch.mean(cross_region / sum_region)
        return 1 - dice


class SolidLoss(nn.Module):
    '''
    General Dice coefficient loss
    '''
    def __init__(self, threshold:float = 10):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, 1]
            target: size [N, 1, H, W, D...]
        '''
        target_solid = torch.flatten(target, start_dim=2)
        target_solid = torch.sum(target_solid, dim=-1)
        target_solid = (target_solid > self.threshold).float()
        solid_loss = torch.mean(-(target_solid*torch.log(predict)+(1-target_solid)*torch.log(1-predict)))
        return solid_loss


class ContainLoss(nn.Module):
    '''
    ContainLoss 
    L(A, B) = aplha *(A * B / A) + (1-alpha)*(A * B / B)
        A: target, B: predict
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.class_index = class_index
        self.n_channel = 2

    def forward(self, predict:Tensor, target: Tensor, alpha:float=0.4):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        assert self.class_index < self.n_channel, 'index beyond output classs'
        predict_reshape = predict.flatten(2).transpose(2, 1)
        class_predict = predict_reshape[:, :, self.class_index]
        class_target = target.flatten(2).transpose(2, 1).squeeze(2)

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region1 = torch.sum(class_target, dim=-1) + self.eps
        sum_region2 = torch.sum(class_predict, dim=-1) + self.eps

        class_dice = torch.mean(cross_region / ((1-alpha)*sum_region1 + \
                                alpha* sum_region2))

        return 1 - class_dice


class ContainLoss2(nn.Module):
    '''
    ContainLoss 
    L(A, B) = aplha *(A * B / A) + (1-alpha)*(A * B / B)
        A: target, B: predict
    '''
    def __init__(self, class_index:int =1, eps:float = 1e-5):
        super().__init__()
        self.eps = eps
        self.class_index = class_index
        self.n_channel = 2

    def forward(self, predict:Tensor, target: Tensor, alpha:float=0.3):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        assert self.class_index < self.n_channel, 'index beyond output classs'
        predict_reshape = predict.flatten(2).transpose(2, 1)
        class_predict = predict_reshape[:, :, self.class_index]
        class_target = target.flatten(2).transpose(2, 1).squeeze(2)

        cross_region = torch.sum(class_predict * class_target, dim=-1) + self.eps
        sum_region1 = torch.sum(class_target, dim=-1) + self.eps
        sum_region2 = torch.sum(class_predict, dim=-1) + self.eps

        class_dice = torch.mean(cross_region / ((1-alpha)*sum_region1 + \
                                alpha* sum_region2))

        return 1 - class_dice



class PyramidLoss(nn.Module):
    '''
    Args:
        n: the number for pyramid,
           should be the len(layers)-1
    '''
    def __init__(self, n: int =3):
        super().__init__()
        self.n = n
        self.kernel_size = 5
        self.contain_loss = nn.ModuleList([ContainLoss(class_index=1,
                                                       alpha=0.2*(i+1)/n) for i in range(n)])
        self.down_sample = nn.AvgPool2d(kernel_size=self.kernel_size, stride=2,
                                        padding=self.kernel_size//2)
        self.threshold = 0.2

    def forward(self, mask_list, target: Tensor):
        loss = []
        temp_target = target.float()

        for i in range(self.n):
            temp_target = self.down_sample(temp_target)
            temp_label = temp_target > self.threshold
            temp_label = temp_label.float()
            loss.append(self.contain_loss[-i-1](mask_list[-i-1], temp_label))

        loss = sum(loss) / self.n
        return loss


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
            target: size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)

        cross_region = torch.sum(predict_reshape * target_onehot, dim=1) + self.eps
        sum_region = torch.sum(predict_reshape + target_onehot, dim=1) + self.eps
        iou = torch.mean(cross_region / (sum_region - cross_region))
        return 1 - iou


class SSLoss(nn.Module):
    '''
    The sensitivity and specificity loss, set sigma for balancing
    '''
    def __init__(self, sigma:float = 0.05, eps:float=1e-5):
        super().__init__()
        self.sigma = sigma
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        n_batch, n_channel = predict.size(0), predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)

        l2_distance = (predict_reshape - target_onehot) ** 2
        sensitivity = torch.sum(l2_distance * target_onehot, dim=1) \
                                / (torch.sum(target_onehot, dim=1) + self.eps)
        specificity = torch.sum(l2_distance * (1 - target_onehot), dim=1) \
                                / (torch.sum(1 - target_onehot, dim=1) + self.eps)
        ssloss = self.sigma * sensitivity + (1 - self.sigma) * specificity
        ssloss = torch.mean(ssloss)
        return ssloss


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
            target: size [N, 1, H, W, D...]
        '''
        n_batch, n_channel = predict.size(0), predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        with torch.no_grad():
            target_onehot = F.one_hot(target_reshape, num_classes = n_channel)

        cross_value = - (1 - predict_reshape) ** self.gamma * \
                        target_onehot * torch.log(predict_reshape)
        focal_loss = torch.mean(cross_value)
        return focal_loss


class RegionCrossEntroLoss(nn.Module):
    '''
    Region focal loss
    Args:
        gamma: the value for balancing difficult samples
    '''
    def __init__(self, eps:float = 1e-9):
        super().__init__()
        self.eps = eps
        self.max_weight = 10
    
    def forward(self, predict:Tensor, target: Tensor, 
                dist:Tensor=None, surface_distance=None):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
            dist: distance maxtrix to surface, size [N, 1, H, W, D...]
        '''
        n_batch, n_channel = predict.size(0), predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        '''
        if dist is not None:
            with torch.no_grad():
                if surface_distance != 0:
                    distance = dist / surface_distance
                    dist_weight = distance**2 + 1
                    dist_weight = dist_weight.flatten(2).transpose(2, 1)
                else:
                    # boundary is defined within 4 pixels
                    dist_weight = 1
                    
                    neighor = 4
                    distance = dist / (surface_distance + neighor)
                    dist_weight = neighor**2 / (distance**2 + 1) + 1
                    dist_weight = dist_weight.flatten(2).transpose(2, 1)
        else:
            dist_weight = 1
        '''
        dist_weight = 1
        cross_value = - target_onehot * torch.log(predict_reshape+self.eps)

        cross_value = cross_value*dist_weight
        cross_loss = torch.mean(cross_value)
        return cross_loss


class CrossEntroLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        n_batch, n_channel = predict.size(0), predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        target_onehot = torch.stack([1-target_reshape, target_reshape], dim=-1)
        '''
        with torch.no_grad():
            target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        '''
        log_predict_reshape = torch.log(torch.clamp(predict_reshape, min=1e-6))
        weight = torch.sum(predict_reshape, dim=1, keepdim=True) + self.eps
        total_sum = torch.sum(target_onehot, dim=(1, 2), keepdim=True)
        weight = (total_sum - weight) / total_sum
        cross_value = - weight*(1-predict_reshape)*target_onehot * log_predict_reshape
        cross_value = torch.mean(cross_value)
        '''
        if torch.any(torch.isnan(cross_value)):
            print('max value')
            print(torch.max(predict_reshape))
            print('min value')
            print(torch.min(predict_reshape))
            print(torch.any(torch.isnan(predict_reshape)))
            print(predict_reshape.shape)
            print('label')
            print(torch.any(torch.isnan(target_onehot)))
            print(torch.any(torch.isinf(target_onehot)))
            print('is inf')
            print(torch.any(torch.isinf(predict_reshape)))
        '''
        return cross_value


class MSEcLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
    
    def forward(self, predict:Tensor, target: Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, 1, H, W, D...]
        '''
        n_channel = predict.size(1)
        predict_reshape = predict.flatten(2).transpose(2, 1)
        target_reshape = target.flatten(2).transpose(2, 1).squeeze(2)
        target_onehot = F.one_hot(target_reshape, num_classes = n_channel)
        return super().forward(predict_reshape.float(), target_onehot.float())

Loss_Dict = {
    'DiceLoss': DiceLoss,
    'DiceClassLoss': DiceClassLoss,
    'BalanceDiceLoss': BalanceDiceLoss,
    'IOULoss': IOULoss,
    'SSLoss': SSLoss,
    'FocalLoss': FocalLoss,
    'CrossEntroLoss': CrossEntroLoss,
    'ContainLoss': ContainLoss,
    'ContainLoss2': ContainLoss2,
    'MSELoss': MSEcLoss,
    'Recall':Recall,
    'Precision': Precision,
    'RecallLoss':RecallLoss,
    'PrecisionLoss': PrecisionLoss,
    'DistributionLoss': DistributionLoss,
    'LocalizationLoss': LocalizationLoss
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