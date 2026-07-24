import torch
import torch.nn.functional as F
from torch import nn


# class DiceLoss:
#     def __init__(self, axis=1, smooth=1e-6, reduction="mean", square_in_union=False):
#         self.axis = axis
#         self.smooth = smooth
#         self.reduction = reduction
#         self.square_in_union = square_in_union
#
#     def __call__(self, pred, targ):
#         targ = self._one_hot(targ, pred.shape[self.axis])
#         assert pred.shape == targ.shape, 'input and target dimensions differ'
#         pred = self.activation(pred)
#
#         sum_dims = list(range(2, len(pred.shape)))
#         inter = torch.sum(pred * targ, dim=sum_dims)
#         union = (torch.sum(pred ** 2 + targ, dim=sum_dims) if self.square_in_union
#                  else torch.sum(pred + targ, dim=sum_dims))
#
#         dice_score = (2. * inter + self.smooth) / (union + self.smooth)
#         loss = 1 - dice_score
#
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
#
#     @staticmethod
#     def _one_hot(x, classes, axis=1):
#         return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)
#
#     def activation(self, x):
#         return F.softmax(x, dim=self.axis)


class DiceLoss:
    def __init__(self, axis=1, smooth=1e-6, reduction="mean", square_in_union=False):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def __call__(self, pred, targ):
        # 将 target 转为 one-hot，并 detach 避免计算图累积
        targ = self._one_hot(targ, pred.shape[self.axis]).float().detach()
        assert pred.shape == targ.shape, 'input and target dimensions differ'

        # 对 pred 做 softmax
        pred = F.softmax(pred, dim=self.axis)

        # sum over H, W (or other spatial dims)
        sum_dims = list(range(2, len(pred.shape)))

        # 交集
        inter = torch.sum(pred * targ, dim=sum_dims)

        # 并集
        if self.square_in_union:
            union = torch.sum(pred ** 2 + targ, dim=sum_dims)
        else:
            union = torch.sum(pred + targ, dim=sum_dims)

        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score

        # reduce
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def _one_hot(x, classes):
        # 使用 F.one_hot，避免循环生成张量
        # x: [B, H, W], 输出: [B, C, H, W]
        one_hot = F.one_hot(x.long(), num_classes=classes)  # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return one_hot


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha: 权重系数，float 或者长度为类别数的list/tensor， None表示不加权
        gamma: 聚焦参数，通常为2
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, float):
                self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        targets = targets.long()  # [B, H, W]
        # 获取前景概率（假设前景类别是 1）
        fg_probs = probs[:, 1, :, :]  # [B, H, W]
        # 错误前景预测：预测值低于 0.5，而真实是前景
        wrong_fg_mask = (fg_probs < 0.5) & (targets == 1)
        wrong_ratio = wrong_fg_mask.sum().float() / (targets == 1).sum().float().clamp(min=1.0)
        # 扁平化
        probs = probs.permute(0, *range(2, inputs.dim()), 1).contiguous()
        probs = probs.view(-1, inputs.size(1))  # [N, C]
        targets = targets.view(-1)  # [N]

        pt = probs[range(probs.shape[0]), targets]  # 取对应类别概率

        log_pt = torch.log(pt + 1e-14)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]
            loss = -at * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2, H, W] - 背景logits(通道0), 前景logits(通道1)
            targets: [B, H, W] - 0=背景, 1=前景
        """
        targets = targets.long()
        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)  # [B, 2, H, W]
        # 获取对应类别的概率
        # 使用gather来正确获取每个目标类别对应的概率
        targets_expanded = targets.unsqueeze(1)  # [B, 1, H, W]
        pt = torch.gather(probs, 1, targets_expanded).squeeze(1)  # [B, H, W]
        # 计算交叉熵损失（与你的CE损失相同的计算方式）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B, H, W]
        # 计算focal loss的调制因子
        focal_factor = (1 - pt) ** self.gamma
        # 计算focal loss
        focal_loss = ce_loss * focal_factor
        # 应用alpha权重
        if self.alpha is not None:
            # 创建alpha权重矩阵
            alpha_weight = torch.ones_like(targets, dtype=torch.float32)
            alpha_weight[targets == 1] = self.alpha
            alpha_weight[targets == 0] = 1 - self.alpha
            focal_loss = alpha_weight * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds shape: [B, C, H, W], targets shape: [B, H, W]
        preds = F.softmax(preds, dim=1)
        preds = preds[:, 1, :, :]  # 二分类，取正类概率
        targets_one_hot = (targets == 1).float()

        intersection = (preds * targets_one_hot).sum(dim=(1, 2))
        union = (preds + targets_one_hot).sum(dim=(1, 2)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou.mean()
        return loss


class Loss():
    def __init__(self, weight=0.1):
        self.dice_loss = DiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 1.1]).cuda())
        # self.focal_loss = FocalLoss(alpha=[0.1, 0.9], gamma=2, reduction='mean')
        # self.focal_loss = BinaryFocalLoss(alpha=0.75, gamma=2)
        # self.iou_loss = IoULoss()
        self.weight = weight

    def __call__(self, pred, targ, epoch):
        ce_loss = self.ce_loss(pred, targ)
        dice_loss = self.dice_loss(pred, targ)
        # iou_loss = self.iou_loss(pred, targ)
        # focal_loss = self.focal_loss(pred, targ)  # 可选，二选一CE or Focal
        # 组合举例
        loss = 0.5*ce_loss+ 0.5*dice_loss
        # loss = dice_loss
        return loss, ce_loss, dice_loss
