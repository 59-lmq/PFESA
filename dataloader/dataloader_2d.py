import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from scipy import ndimage


class Dataset2D(Dataset):
    def __init__(self, data_path, txt_dict, split='train', num=None,
                 resize_shape=(256, 256), mean=None, std=None,
                 ):
        self.data_path = data_path
        self.txt_dict = txt_dict

        self.resize_shape = resize_shape
        self.mean = mean
        self.std = std

        with open(self.txt_dict[split], 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'image', self.image_list[idx])
        lab_path = img_path.replace('image', 'label')

        img = Image.open(img_path)
        lab = Image.open(lab_path)

        img = np.array(img)
        lab = np.array(lab)

        # Augmentation
        img, lab = resize_2d(img, lab, size=self.resize_shape)
        img, lab = random_rot_flip(img, lab)
        img, lab = random_rotate(img, lab)

        # Normalize
        img = normalize_2d(img, self.mean, self.std)

        # binary
        lab[lab>0] = 1

        # to tensor
        img, lab = to_tensor_2d(img, lab)
        sample = {'image': img, 'label': lab, 'name': self.image_list[idx]}

        return sample


def normalize_2d(img, mean, std, max_pixel_value=255):
    assert max_pixel_value > 0
    return (img/max_pixel_value - mean) / std


def resize_2d(image, label, size=(256, 256)):
    if label is None:
        image = Image.fromarray(image)
        image = image.resize(size)
        return np.array(image)
    # 如果image和label的shape 与 size是一致的，则不进行操作，返回原始的image和label
    if image.shape[:2] == size and label.shape[:2] == size:
        return image, label
    image = Image.fromarray(image)
    label = Image.fromarray(label)

    image = image.resize(size)
    label = label.resize(size)

    return np.array(image), np.array(label)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(20, 80)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def to_tensor_2d(x, y):
    x = x.transpose(2, 0, 1).astype(np.float32)

    # to tensor
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()

    return x, y


def dataset_2d_test():

    data_path = r'../ISIC-2017'

    txt_dict = {
        'train': os.path.join(data_path, 'train.txt'),
        'val': os.path.join(data_path, 'val.txt'),
        'test': os.path.join(data_path, 'test.txt'),
    }

    dataset = Dataset2D(data_path, txt_dict, split='train', num=10,
                        resize_shape=(256, 256), mean=[0.787803, 0.512017, 0.784938], std=[0.428206, 0.507778, 0.426366])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # show data

    for idx, data in enumerate(dataloader):

        img = data['image']
        lab = data['label']
        name = data['name']

        print(f' type of image: {type(img)}, type of label: {type(lab)}')
        print(f' range of image and label: {img.min(), img.max(), lab.min(), lab.max()}')

        print(f'idx: {idx}, img: {img.shape}, lab: {lab.shape}, name: {name}')

        if idx == 0:
            break

    print('Done!')


if __name__ == '__main__':
    dataset_2d_test()
