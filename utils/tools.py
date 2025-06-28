import os
import time

import h5py
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from skimage import measure


def make_dir(file_path):
    """
    Create a directory if not exist.
    :param file_path: the path of the directory to be created.
    :return: None
    """

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    return None


def print_args(args):
    """
    Print arguments.
    :param args:
    :return:
    """
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:\t{}\n".format(arg, content)
    return s


def rescale(image_label, w_ori, h_ori, z_ori, flag):
    """
    Resize label map (int)
    :param image_label: label or image map
    :param w_ori: original width
    :param h_ori: original height
    :param z_ori: original depth
    :param flag: interpolation method, 'trilinear' or 'nearest'
    :return: resized label or image map
    """
    # resize label map (int)
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = np.zeros((w_ori, h_ori, z_ori))
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id])
            image_label_bn = torch.from_numpy(image_label_bn.astype(float))
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori),
                                                             mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :].numpy()
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori

    if flag == 'nearest':
        image_label = torch.from_numpy(image_label.astype(float))
        image_label = image_label[None, None, :, :, :]
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].numpy()
    return image_label


def image_normalise(image, norm_type):
    """
    Normalise image to [0, 1] or [-1, 1]
    :param image: image to be normalised
    :param norm_type:
        type 1, Rescaling, min-max Normalization, range [0, 1] : (image - min(image)) / (max(image) - min(image))
        type 2, Standardization, z-score Normalization : (image - mean(image)) / std(image)
        type 3, range [0, 1] : (image - 500) / (2500 - 500)
        type 4, mean Normalization : (image - mean(image)) / (max(image) - min(image))
        type 5, range [0, 1] : (image - 0) / (2500 - 0)
    :return: normalised image
    """
    if norm_type == 1:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif norm_type == 2:
        image = (image - np.mean(image)) / np.std(image)
    elif norm_type == 3:
        image[image < 500] = 500
        image[image > 2500] = 2500
        image = (image - 500) / (2500 - 500)
    elif norm_type == 4:
        image = (image - np.mean(image)) / (np.max(image) - np.min(image))
    elif norm_type == 5:
        image[image < 0] = 0
        image[image > 2500] = 2500
        image = (image - 0) / (2500 - 0)

    return image


def read_nii_data(nii_path):
    """
    读取 nii 数据
    :param nii_path: the path of nii data
    :return: nii data, affine, spacing
    """
    img = nib.load(nii_path)
    data = img.get_fdata()
    return data, img.affine, img.header['pixdim'][1:4]


def save_nii_data(data, affine, save_path):
    """
    保存 nii 数据
    :param data: nii data
    :param affine: nii affine
    :param save_path: the path of nii data to be saved
    :return: None
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, save_path)
    return None


def read_nii_image_data(data_path_img, default_spacing=0.3, is_rescale=False, norm_type=1):
    """
    读取 image 数据
    :param data_path_img: the path of image data
    :param default_spacing: the default spacing of image data，插值后的数据的 spacing，默认为 0.3
    :param is_rescale: whether the data is rescaled，是否需要进行插值，默认为 True
    :param norm_type: the type of normalisation，归一化的类型，默认为 1
        see function image_normalise
        type 1: (image - min(image)) / (max(image) - min(image))
        type 2: (image - mean(image)) / std(image)
        type 3: (image - 500) / (2500 - 500)
    :return: image data, affine, spacing, shape
    """
    images, affine, spacing = read_nii_data(data_path_img)
    w, h, d = images.shape
    # print(images.shape)
    whether_spacing_is_same = int(float(spacing[0] * 10)) == int(10 * default_spacing)

    # w * (spacing[0] / 0.4) -> 是指把原始数据的spacing 变成 0.4 mm
    if is_rescale and not whether_spacing_is_same:
        images = rescale(images,
                         w * (spacing[0] / default_spacing),
                         h * (spacing[0] / default_spacing),
                         d * (spacing[0] / default_spacing), 'nearest')
    images = image_normalise(images, norm_type=norm_type)

    return images, affine, spacing, (w, h, d)


def read_nii_label_data(data_path_lab, default_spacing=0.3, is_rescale=False):
    """
    读取 label 数据
    :param data_path_lab: the path of label data
    :param default_spacing: the default spacing of image data，插值后的数据的 spacing，默认为 0.3
    :param is_rescale: whether the data is rescaled，是否需要进行插值，默认为 True
    :return: label data, label affine
    """
    labels, affine, spacing = read_nii_data(data_path_lab)
    w, h, d = labels.shape
    # print(labels.shape)
    whether_spacing_is_same = int(float(spacing[0] * 10)) == int(10 * default_spacing)

    # w * (spacing[0] / 0.4) -> 是指把原始数据的spacing 变成 0.4 mm
    if is_rescale and not whether_spacing_is_same:
        labels = rescale(labels,
                         w * (spacing[0] / default_spacing),
                         h * (spacing[0] / default_spacing),
                         d * (spacing[0] / default_spacing), 'trilinear')

    return labels, affine


def read_data(data_path_img, data_patch_lab, is_rescale=True, default_spacing=0.3, norm_type=1):
    """
    同时读取 image 和 label 数据
    :param data_path_img: the path of image data
    :param data_patch_lab: the path of label data
    :param is_rescale: whether the data is rescaled，是否需要进行插值，默认为 True
    :param default_spacing: the default spacing of image data，插值后的数据的 spacing，默认为 0.3
    :param norm_type: the type of normalisation，归一化的类型，默认为 1
        type 1: (image - min(image)) / (max(image) - min(image))
        type 2: (image - mean(image)) / std(image)
        type 3: (image - 500) / (2500 - 500)
    :return: image and label data
    """
    images, _, _, _ = read_nii_image_data(data_path_img, default_spacing, is_rescale=is_rescale, norm_type=norm_type)
    labels, _ = read_nii_label_data(data_patch_lab, default_spacing, is_rescale=is_rescale)

    return images, labels


def load_h5(file_path):
    """
    Load h5 file.
    :param file_path: path of the h5 file.
    :return: data and label.
    """
    with h5py.File(file_path, 'r') as f:
        image = f['image'][:]
        label = f['label'][:]
    return image, label