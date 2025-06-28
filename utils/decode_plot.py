import matplotlib.pyplot as plt
import numpy as np
import torch


def get_tooth_labels():
    """
    Load the mapping that tooth classes with label colors
    Returns:
        np.ndarray with dimensions (2, 3)
    """
    return np.asarray([[0, 0, 0], [0, 64, 128]])


def get_ab_labels():
    """
    Load the mapping that tooth classes with label colors
    Returns:
        np.ndarray with dimensions (3, 3)
    """
    return np.asarray([[0, 0, 0], [0, 64, 128], [0, 128, 64]])


def decode_seg_map_sequence(label_masks, dataset='tooth'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_seg_map(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_seg_map(label_mask, dataset_stage, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        dataset_stage (str): the name of the dataset
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset_stage == 'tooth':
        n_classes = 2
        label_colours = get_tooth_labels()
    elif dataset_stage == 'ab':
        n_classes = 3
        label_colours = get_ab_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
