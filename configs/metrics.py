
import numpy as np
import torch
from medpy import metric


def calculate_metric(pred, gt, affine=None):
    """
    计算指标，返回 dice, jc, asd, sen, hd95
    :param pred:  预测图
    :param gt:  真实标签
    :param affine:  仿射矩阵
    :return:  dice, jc, asd, sen, hd95
    """

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    if pred.sum() == 0 or gt.sum() == 0:
        dice = 0
        jc = 0
        asd = 50
        sen = 0
        hd95 = 50
    else:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        if affine is None:
            asd = metric.binary.asd(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
        else:
            asd = metric.binary.asd(pred, gt, voxelspacing=abs(affine))
            hd95 = metric.binary.hd95(pred, gt, voxelspacing=abs(affine))
        sen = metric.binary.sensitivity(pred, gt)

    return dice, jc, asd, sen, hd95


def only_dice(pred, gt):
    """
    计算 Dice
    :param pred:  预测图
    :param gt:  真实标签
    :return:  dice
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    if pred.sum() == 0 or gt.sum() == 0:
        dice = 0
    else:
        dice = metric.binary.dc(pred, gt)
    return dice