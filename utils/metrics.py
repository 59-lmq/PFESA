import torch
from medpy import metric
import numpy as np
from ptflops import get_model_complexity_info	## 导入ptflops模块


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


def cal_params_flops(model, logger=None, input_size=(3, 256, 256)):
    """
    计算模型的参数量和计算量
    :param model:  需要计算的模型
    :param logger:  日志 logger
    :param input_size:  输入数据的shape，如 torch.randn(1, 3, 224, 224) or torch.randn(1, 1, 128, 128, 128)
    :return:
    """
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False, verbose=True)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total/1e6))
    print_str = f'FLOPs (MACs): {macs}, Trainable params: {params}, Total params: : {total/1e6:.4f}'
    if logger is not None:
        logger.info(print_str)
    else:
        print(print_str)


class AverageMetricMore(object):
    """
    计算多个指标的平均值和标准差
    e.g.
    metric = AverageMetricMore()
    metric.update('dice', 0.9)
    metric.update('dice', 0.8)
    metric.update('dice', 0.7)
    metric.update('jc', 0.9)
    metric.update('jc', 0.8)
    metric.update('jc', 0.7)

    mean, std, metrics = metric.get_all()
    print(mean)
    print(std)
    print(metrics)

    # {'dice': 0.8, 'jc': 0.8}
    # {'dice': 0.08164965809277261, 'jc': 0.08164965809277261}
    # {'dice': [0.9, 0.8, 0.7], 'jc': [0.9, 0.8, 0.7]}
    """

    def __init__(self):
        self.std = {}
        self.mean = {}
        self.metrics = {}

    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_mean(self):
        for metric_name, values in self.metrics.items():
            self.mean[metric_name] = np.mean(values)

    def get_std(self):
        for metric_name, values in self.metrics.items():
            self.std[metric_name] = np.std(values)

    def get_all(self):
        self.get_std()
        self.get_mean()
        return self.mean, self.std, self.metrics


