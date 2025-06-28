# Description: This file contains utility functions for the project.
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import torch



def save_img(img, msk, msk_pred, data_idx, save_path, mean=None, std=None, id_name=None):
    """
    Save image, mask, and predicted mask.
    :param img: [1, C, H, W]
    :param msk: [1, H, W]
    :param msk_pred: [1, H, W]
    :param data_idx:  index of the image
    :param save_path:  save path. e.g., 'outputs/', 'outputs/val/' always with '/', is a directory
    :param mean:  mean value of the image
    :param std:  standard deviation of the image
    :param id_name:  id name of the image
    :return:
    """
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, C]
    msk = msk.squeeze(0).detach().cpu().numpy()  # [H, W]
    msk_pred = msk_pred.squeeze(0).detach().cpu().numpy()  # [H, W]

    # reverse normalization
    if mean is not None and std is not None:
        img = img * std + mean

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if id_name is not None:
        # save_path = save_path + test_data_name + '_'
        save_name = os.path.join(save_path, str(id_name) + '.png')
    else:
        save_name = os.path.join(save_path, str(data_idx) + '.png')
    plt.savefig(save_name, dpi=300)
    plt.close()

    return save_name


def save_pred(pred, save_path, id_name):
    """
    Save predicted mask.
    :param pred: [1, H, W]
    :param save_path:  save path. e.g., 'outputs/', 'outputs/val/' always with '/', is a directory
    :param id_name:  id name of the image
    :return:
    """
    pred = pred.detach().cpu().numpy()  if isinstance(pred, torch.Tensor) else pred
    pred = np.squeeze(pred, axis=0)  # [H, W]

    save_name = os.path.join(save_path, str(id_name) + '.png')
    Image.fromarray(pred).save(save_name)

    return save_name