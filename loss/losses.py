import torch
import numpy as np

from torch.nn import functional as F


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def train_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def to_one_hot_label(label, num_classes, device):

    size_len = len(label.size())
    if size_len == 4:  # 3D
        n, x, y, z = label.size()  # batch_size, h, w, d
        one_hot_super = torch.zeros(n, num_classes, x, y, z).to(device)
        one_hot_super = one_hot_super.scatter_(1, label.view(n, 1, x, y, z), 1)
        return one_hot_super.float()
    elif size_len == 3:  # 2D
        n, x, y = label.size()  # batch_size, h, w
        one_hot_super = torch.zeros(n, num_classes, x, y).to(device)
        one_hot_super = one_hot_super.scatter_(1, label.view(n, 1, x, y), 1)
        return one_hot_super.float()


def dice_loss_multi(score, target, device, is_one_hot=False):
    loss = 0
    class_num = score.shape[1]
    if is_one_hot:
        target = to_one_hot_label(target, class_num, device)
    for i in range(class_num):
        if i == 0:
            continue
        loss += dice_loss(score[:, i], target[:, i])
    return loss / class_num
