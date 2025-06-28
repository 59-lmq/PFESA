import math
import os
import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm


def get_largest_class(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largest_class = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_class


def get_time_stamp():
    import time
    import datetime
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    return st


def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('../LA/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                      image_list]
    elif dataset_name == "Pancreas_CT":
        with open('../Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    elif dataset_name == "Tooth":
        with open('../Tooth/valid.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../Tooth/tooth_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        images = h5f['image'][:]
        labels = h5f['label'][:]
        prediction, score_map = test_single_case_first_output(model, images, stride_xy, stride_z, patch_size,
                                                              num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, labels)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18,
                  stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0,
                  abb_name='', dataset_name='LA'):
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    all_metric = []
    all_metrics = []
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        if dataset_name == 'LA':
            images = h5f['image'][:]
            labels = h5f['label'][:]
        elif dataset_name == 'Tooth':
            images = h5f['images'][:]
            labels = h5f['labels'][:]

        if preproc_fn is not None:
            images = preproc_fn(images)
        prediction, score_map = test_single_case_first_output(model, images, stride_xy, stride_z, patch_size,
                                                              num_classes=num_classes)
        if num_outputs > 1:
            prediction_average, score_map_average = test_single_case_average_output(model, images, stride_xy, stride_z,
                                                                                    patch_size, num_classes=num_classes)
        if nms:
            prediction = get_largest_class(prediction)
            if num_outputs > 1:
                prediction_average = get_largest_class(prediction_average)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
            if num_outputs > 1:
                single_metric_average = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_per_case(prediction, labels[:])
            if num_outputs > 1:
                single_metric_average = calculate_metric_per_case(prediction_average, labels[:])

        if metric_detail:
            all_metric.append([ith] + list(single_metric))
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
                ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
            if num_outputs > 1:
                all_metrics.append([ith] + list(single_metric_average))
                print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
                    ith, single_metric_average[0], single_metric_average[1], single_metric_average[2],
                    single_metric_average[3]))

        total_metric += np.asarray(single_metric)
        if num_outputs > 1:
            total_metric_average += np.asarray(single_metric_average)

        if save_result:
            pred_save_name = os.path.join(test_save_path, '%02d_pred.nii.gz' % ith)
            scores_save_name = os.path.join(test_save_path, '%02d_scores.nii.gz' % ith)
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), pred_save_name)
            # nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), scores_save_name)

            img_save_name = os.path.join(test_save_path, '%02d_img.nii.gz' % ith)
            gt_save_name = os.path.join(test_save_path, '%02d_gt.nii.gz' % ith)
            nib.save(nib.Nifti1Image(images[:].astype(np.float32), np.eye(4)), img_save_name)
            nib.save(nib.Nifti1Image(labels[:].astype(np.float32), np.eye(4)), gt_save_name)

        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is decoder 1 {}'.format(avg_metric))
    if num_outputs > 1:
        avg_metric_average = total_metric_average / len(image_list)
        print('average metric of all decoders is {}'.format(avg_metric_average))

    metric_txt_name = os.path.join(test_save_path, '{}_{}_performance.txt'.format(abb_name, model_name))
    with open(metric_txt_name, 'w') as f:
        f.writelines('metrics: dice, jc, hd, asd \n')
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))
        for m_ in all_metric:
            f.writelines('%d, %.5f, %.5f, %.5f, %.5f \n' % (m_[0], m_[1], m_[2], m_[3], m_[4]))
        if num_outputs > 1:
            f.writelines('average metric of all decoders is {} \n'.format(avg_metric_average))
            for m_ in all_metric:
                f.writelines('%d, %.5f, %.5f, %.5f, %.5f \n' % (m_[0], m_[1], m_[2], m_[3], m_[4]))
    return avg_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    ps = patch_size

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, ((wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)), mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        if len(y[0]) > 1:
                            y = y[0][0]
                        else:
                            y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += y
                cnt[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int8)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def test_single_case_average_output(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    ps = patch_size

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, ((wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)), mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = net(test_patch)
                    num_outputs = len(y_logit)
                    y = torch.zeros(y_logit[0].shape).cuda()
                    for idx in range(num_outputs):
                        y += y_logit[idx]
                    y /= num_outputs

                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += y
                cnt[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int8)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def calculate_metric_per_case(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd