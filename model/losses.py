import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        # weight: tensor of size [num_classes] (e.g., [1.5, 10.0, 1.0, 2.0])
        self.weight = weight

    def forward(self, pred, target):
        # pred: [batch_size, num_classes, H, W] (logits)
        # target: [batch_size, H, W] (class indices)
        return nn.CrossEntropyLoss(weight=self.weight)(pred, target)
    
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon  

    def forward(self, pred, target):
        # pred: [batch_size, num_classes, H, W] (logits)
        # target: [batch_size, H, W] (class indices)
        pred = torch.softmax(pred, dim=1)  
        # one-hot encode target
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)

        # denominator (sum over batch, H, W)
        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))  
        # numerator (sum over batch, H, W)
        union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return 1 - dice.mean()
    
class CEandDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.7, dice_weight=0.3, ce_class_weights=None):
        super(CEandDiceLoss, self).__init__()
        self.ce_loss = WeightedCrossEntropyLoss(weight=ce_class_weights)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.alpha = alpha  # Optional class weights (tensor of [num_classes])
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, pred, target):
        # pred: [batch_size, num_classes, H, W] (logits)
        # target: [batch_size, H, W] (class indices)

        # Compute softmax probabilities
        pred_prob = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
        
        # One-hot encode target
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = (pred_prob * target_one_hot).sum(dim=1, keepdim=True)  # Probability of true class
        focal_weight = (1 - p_t).pow(self.gamma)  # [batch_size, 1, H, W]

        # Compute log probability
        log_p_t = torch.log(pred_prob + 1e-8) * target_one_hot  # Add small epsilon to avoid log(0)
        loss = -focal_weight * log_p_t.sum(dim=1)  # [batch_size, H, W]

        # Apply alpha weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[target]  # [batch_size, H, W]
            loss = loss * alpha_t

        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss