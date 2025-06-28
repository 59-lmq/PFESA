import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader.dataloader_3d import Tooth, RandomCrop, RandomRotFlip, ToTensor, LAHeart
from models.net_factorys import get_network_3d
from loss.losses import dice_loss
from utils.metrics import AverageMetricMore
from utils.decode_plot import decode_seg_map_sequence
from configs.dataset_cfg import dataset_cfg

from utils.val_patch import test_all_case


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='./results')
    parser.add_argument('--dataset_name', default='LA', help='LA, Tooth')
    parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')

    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--patch_size', type=int, default=(112, 112, 80), help='patch_size for input data')
    parser.add_argument('--num_class', type=int, default=2, help='class of you want to segment')
    parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

    parser.add_argument('-n', '--network', default='unet3d', type=str,
                        help='unet3d, vnet, unetr_pp')
    parser.add_argument('-a', '--attention', default='CBAM', type=str,
                        help='CBAM, SE, ECA, SimAM, SIAM, Identity, PFESA')

    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--num_workers', type=int, default=4, help='num-workers to use')
    args_ = parser.parse_args()

    return args_


def init_seeds(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size * len(args.gpu.split(','))
    max_iterations = args.max_iterations
    base_lr = args.base_lr

    init_seeds(args.seed)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    base_path_dataset = {
        'GlaS': r'../GlaS',
        'ISIC-2017': r'../ISIC-2017',
        'LA': r'../LA',
        'Tooth': r'../Tooth'
    }

    path_dataset = base_path_dataset[dataset_name]
    patch_size = cfg['PATCH_SIZE']

    exp_name = (f"{args.network}-a={args.attention}-l={args.base_lr}-e={args.max_iterations}"
                f"-b={args.batch_size}")

    path_trained_models = os.path.join(args.path_trained_models, os.path.split(path_dataset)[1])
    os.makedirs(path_trained_models, exist_ok=True)

    path_trained_models = os.path.join(path_trained_models, exp_name)
    os.makedirs(path_trained_models, exist_ok=True)

    # log path
    log_path = os.path.join(path_trained_models, 'log/train_')

    logger.add(log_path + '{time}.txt', rotation='00:00')
    writer = SummaryWriter(log_dir=log_path + f'run_{args.max_iterations}')

    logger.info(f'Train Begin！')
    logger.info(f' args: {args}')

    st_time = time.time()

    net = get_network_3d(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'],
                           attention_type=args.attention, img_size=patch_size)
    logger.info(f'Network: {net}')
    net.cuda()

    if dataset_name == 'LA':
        dataset_train = LAHeart(path_dataset,
                                split='train',
                                transform=transforms.Compose([
                                    RandomCrop(patch_size),
                                    RandomRotFlip(),
                                    ToTensor(),
                                ]))
        data_test = LAHeart(path_dataset,
                              split='test',
                              transform=transforms.Compose([
                                  RandomCrop(patch_size),
                                  RandomRotFlip(),
                                  ToTensor(),
                              ]))
    elif dataset_name == 'Tooth':
        dataset_train = Tooth(path_dataset,
                              split='train',
                              transform=transforms.Compose([
                                  RandomCrop(patch_size),
                                  RandomRotFlip(),
                                  ToTensor(),
                              ]))
        data_test = Tooth(path_dataset,
                              split='valid',
                              transform=transforms.Compose([
                                  RandomCrop(patch_size),
                                  RandomRotFlip(),
                                  ToTensor(),
                              ]))

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.99, weight_decay=0.0001)

    logger.info(f'每个epoch迭代 {len(train_loader)} 次')

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = base_lr

    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time_1 = time.time()
        epoch_loss = 0
        for idx_batch, sample_batch in enumerate(train_loader):
            time_2 = time.time()
            image_batch, label_batch = sample_batch['image'], sample_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = net(image_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)

            loss = 0.5 * (loss_ce + loss_dice)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 500 == 0:
                image = image_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                grid_image = make_grid(decode_seg_map_sequence(image), 5, normalize=False)
                writer.add_image('train/Prediction', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1).data.cpu().numpy()
                grid_image = make_grid(decode_seg_map_sequence(image), 5, normalize=False)
                writer.add_image('train/GroundTruth', grid_image, iter_num)

            logger.info(f' iteration {iter_num} : loss : {loss.item()} , dice_acc : {1 - loss_dice.item()}, '
                        f'time: {time.time() - time_2} s')
            if iter_num > max_iterations:
                break
        logger.info(f'epoch {epoch_num} | loss: {epoch_loss / len(train_loader)}, time: {time.time() - time_1} s')

        # Validation
        net.eval()
        val_epoch_loss = 0
        val_metrics = AverageMetricMore()
        time_3_1 = time.time()
        for idx_batch, sample_batch in enumerate(test_loader):
            time_3 = time.time()
            image_batch, label_batch = sample_batch['image'], sample_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = net(image_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)

            val_loss = 0.5 * (loss_ce + loss_dice)
            val_epoch_loss += val_loss.item()

            # 计算指标
            dice = 1 - loss_dice.item()

            val_metrics.update('dice', dice)

            logger.info(f' val stage | iteration {idx_batch}: loss : {val_loss.item()} , dice_acc : {dice},'
                        f'time: {time.time() - time_3} s')
        del image_batch, label_batch, loss_ce, outputs_soft, loss_dice, val_loss

        val_metrics.get_all()
        val_average_dice = val_metrics.mean['dice']
        if val_average_dice > best_dice:
            best_dice = val_average_dice
            save_model_path = os.path.join(path_trained_models, 'iter_best.pth')
            torch.save(net.state_dict(), save_model_path)
            logger.info(f" best dice:{best_dice}, save model to {save_model_path} !")

        logger.info(f'epoch {epoch_num} | val loss: {val_epoch_loss / len(test_loader)}, '
                    f'time: {time.time() - time_3_1} s, '
                    f'mean metrics: {val_metrics.mean}, std metrics: {val_metrics.std}')
        net.train()

        writer.add_scalars('loss/epoch_loss',
                           {'train': epoch_loss / len(train_loader),
                            'val': val_epoch_loss / len(test_loader)}, epoch_num)

        if iter_num > max_iterations:
            break
    save_model_path = os.path.join(path_trained_models, 'final.pth')
    torch.save(net.state_dict(), save_model_path)
    logger.info(f"save model to {save_model_path} using time: {(time.time() - st_time)/3600} h")
    writer.close()


    # Test
    t_time = time.time()
    net.load_state_dict(torch.load(os.path.join(path_trained_models, 'final.pth')))

    nii_save_path = os.path.join(path_trained_models, 'test_nii')
    os.makedirs(nii_save_path, exist_ok=True)

    avg_metric = (0, 0, 0, 0)
    if dataset_name == 'LA':
        with open(os.path.join(path_dataset, 'test.list'), 'r') as f:
            test_list = f.readlines()
        image_list = [path_dataset + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                  test_list]

        avg_metric = test_all_case(args.network,
                                   1, net, image_list, num_classes=cfg['NUM_CLASSES'],
                                   patch_size=patch_size, stride_xy=18, stride_z=4,
                                   save_result=True, test_save_path=nii_save_path,
                                   metric_detail=1, nms=0,
                                   abb_name=args.attention,
                                  dataset_name='LA')
    elif dataset_name == 'Tooth':
        with open(os.path.join(path_dataset, 'valid.txt'), 'r') as f:
            test_list = f.readlines()
        image_list = [path_dataset + "/tooth_h5/" + item.replace('\n', '') + "" for item in
                  test_list]

        avg_metric = test_all_case(args.network,
                                   1, net, image_list, num_classes=cfg['NUM_CLASSES'],
                                   patch_size=patch_size, stride_xy=64, stride_z=48,
                                   save_result=True, test_save_path=nii_save_path,
                                   metric_detail=1, nms=0,
                                   abb_name=args.attention,
                                  dataset_name='Tooth')

    logger.info(f'\nDice: {avg_metric[0]:.4f}'
                f'\nJC: {avg_metric[1]:.4f}'
                f'\nHD95: {avg_metric[2]:.4f}'
                f'\nASD: {avg_metric[3]:.4f}')

    logger.info(
        f'network: {args.network}, attention: {args.attention}, dataset: {dataset_name}， test mean dice: {avg_metric[0]:.4f}')
    logger.info(f'Test time: {time.time() - t_time} s')


if __name__ == '__main__':
    main()
