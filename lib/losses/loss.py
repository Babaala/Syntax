"""
This part is the available loss function
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)  # Not include softmax
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)  

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets): # 包含了log_softmax函数，调用时网络输出层不需要加log_softmax
        return self.nll_loss((1 - F.softmax(inputs,1)) ** self.gamma * F.log_softmax(inputs,1), targets)

class FocalLoss_t(nn.Module):
    def __init__(self, num_classes, extra_weight=None):
        super(FocalLoss_t, self).__init__()
        self.num_classes = num_classes
        self.extra_weights = torch.tensor([0,0]).cuda()     # 手动添加不同类别的权重;可以self.extra_weights=extra_weight来通过上层传递参数
    
    def forward(self, input, target):
        '''
        :param input: shape [batch_size,num_classes,H,W] 仅仅经过卷积操作后的输出，并没有经过任何激活函数的作用
        :param target: shape [batch_size,H,W]
        :return:
        '''

    
        n, c, h, w = input.size()
        
        target = target.long()
        input = input.contiguous().view(-1, c)
        target = target.contiguous().view(-1)
        

        numbers = [torch.sum(target == i).item() for i in range(self.num_classes)]    
        
        frequency = torch.tensor(numbers, dtype=torch.float32)
        frequency = frequency.numpy()
        classWeights = self.compute_class_weights(frequency)
        '''
        根据当前给出的ground truth label计算出每个类别所占据的权重
        '''
        
        weights = torch.from_numpy(classWeights).float().cuda()
        # # 人为添加权重
        # print("weights:", weights)
        weights += weights + self.extra_weights

        focal_frequency = F.nll_loss(F.softmax(input, dim=1), target, reduction='none')
        focal_frequency += 1.0#shape [num_samples] 1-P（gt_classes）
        focal_frequency = torch.pow(focal_frequency, 2) # torch.Size([75])
        focal_frequency = focal_frequency.repeat(c, 1)
        '''
        repeat操作后，focal_frequency shape [num_classes,num_samples]
        '''
        focal_frequency = focal_frequency.transpose(1, 0)
        loss = F.nll_loss(focal_frequency * (torch.log(F.softmax(input, dim=1))), target, weight=weights,
                reduction='mean')
        return loss

    def compute_class_weights(self, histogram):
        classWeights = np.ones(self.num_classes, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(self.num_classes):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        
        n, h, w = target.shape      # 这里的target是[batch_size,h,w]的，input是[batch_size, num_classes, h, w]的，所以需要将target处理一下
        target_1 = target.clone().reshape(n,1,h,w)
        target_0 = target.clone().reshape(n,1,h,w)
        target_0 = 1-target_0
        # 然后进行拼接
        target = torch.cat((target_0, target_1), dim=1)


        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
