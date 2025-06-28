import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import time
import os
import numpy as np
import random
import pandas
from loguru import logger

from configs.dataset_cfg import dataset_cfg
from configs.warmup import GradualWarmupScheduler
from models.net_factorys import get_network
from dataloader.dataloader_2d import Dataset2D
from utils.metrics import only_dice, calculate_metric
from loss.losses import dice_loss_multi
from utils.util import save_img, save_pred

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='./results')
    parser.add_argument('--dataset_name', default='GlaS', help='ISIC-2017, GlaS')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.1, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('-n', '--network', default='unet', type=str)
    parser.add_argument('-r', '--radius', default=0.1, type=float, help='radius for Gaussian Filter')
    parser.add_argument('-a', '--attention', default='PFESA', type=str,
                        help='CBAM, SE, ECA, SimAM, SIAM, Identity, PFESA')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--num_workers', type=int, default=4, help='num-workers to use')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    init_seeds(args.seed)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    base_path_dataset = {
        'GlaS': r'../GlaS',
        'ISIC-2017': r'../ISIC-2017'
    }

    path_dataset = base_path_dataset[dataset_name]

    exp_name = (f"{args.network}-a={args.attention}-r={args.radius}-l={args.lr}-e={args.num_epochs}-s={args.step_size}-"
                f"g={args.gamma}-b={args.batch_size}-w={args.warm_up_duration}")

    # path_trained_models = path_trained_models + '/' + dataset_name
    path_trained_models = os.path.join(args.path_trained_models, os.path.split(path_dataset)[1])
    os.makedirs(path_trained_models, exist_ok=True)

    # path_trained_models = args.path_trained_models + '/' + dataset_name '/' + exp_name
    path_trained_models = os.path.join(path_trained_models, exp_name)
    os.makedirs(path_trained_models, exist_ok=True)

    # log path
    log_path = os.path.join(path_trained_models, 'log/train_')

    logger.add(log_path + '{time}.txt', rotation='00:00')
    writer = SummaryWriter(log_dir=log_path + f'run_{args.num_epochs}')

    txt_dict = {
        'train': os.path.join(path_dataset, 'train.txt'),
        'val': os.path.join(path_dataset, 'val.txt'),
        'test': os.path.join(path_dataset, 'test.txt')
    }

    dataset_train = Dataset2D(path_dataset, txt_dict, split='train', num=None,
                              resize_shape=cfg['RESIZE'], mean=cfg['MEAN'], std=cfg['STD'])
    dataset_val = Dataset2D(path_dataset, txt_dict, split='val', num=None,
                            resize_shape=cfg['RESIZE'], mean=cfg['MEAN'], std=cfg['STD'])

    dataset_test = Dataset2D(path_dataset, txt_dict, split='test', num=None,
                             resize_shape=cfg['RESIZE'], mean=cfg['MEAN'], std=cfg['STD'])

    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'val': DataLoader(dataset_val, batch_size=1, shuffle=False,
                          pin_memory=True, num_workers=args.num_workers),
        'test': DataLoader(dataset_test, batch_size=1, shuffle=False,
                           pin_memory=True, num_workers=args.num_workers)
    }

    num_batches = {'train': len(dataloaders['train']), 'val': len(dataloaders['val']), 'test': len(dataloaders['test'])}

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'],
                        attention_type=args.attention)
    model = model.to(device)

    logger.info(f'Args: \n')
    for arg, value in vars(args).items():
        logger.info(f'{arg} : {value}')
    logger.info(f'网络结构: \n{model}')

    # Training Strategy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5 * 10 ** args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                              total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)

    # Train & Val
    since = time.time()
    best_dice = 0.0
    best_epoch = 0
    final_dice = 0.0

    for epoch in range(args.num_epochs):
        e_t_time = time.time()
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        loss_ce_num = 0.0
        loss_dice_num = 0.0

        for i, data in enumerate(dataloaders['train']):
            inputs_train = Variable(data['image'].to(device))
            mask_train = Variable(data['label'].to(device))

            outputs_train = model(inputs_train)

            # print(outputs_train.shape, mask_train.shape)

            loss_ce = F.cross_entropy(outputs_train, mask_train)
            outputs_train_soft = F.softmax(outputs_train, dim=1)  # batch_size, num_classes, H, W
            loss_dice = dice_loss_multi(outputs_train_soft, mask_train, device, is_one_hot=True)

            loss_train = loss_ce + loss_dice

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            train_loss += loss_train.item()
            loss_ce_num += loss_ce.item()
            loss_dice_num += loss_dice.item()

        scheduler_warmup.step()
        logger.info(f'Epoch {epoch}/{args.num_epochs - 1} Train Loss: {train_loss / num_batches["train"]}'
                    f' Time: {time.time() - e_t_time} s')

        writer.add_scalar('Loss/train', train_loss / num_batches['train'], epoch)
        writer.add_scalar('Loss_CE/train', loss_ce_num / num_batches['train'], epoch)
        writer.add_scalar('Loss_Dice/train', loss_dice_num / num_batches['train'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        e_v_time = time.time()

        # Validation
        with torch.no_grad():
            model.eval()
            dice_epoch = 0.0

            for i, data in enumerate(dataloaders['val']):
                inputs_val = Variable(data['image'].to(device))
                mask_val = Variable(data['label'].to(device))
                outputs_val = model(inputs_val)

                loss_ce = F.cross_entropy(outputs_val, mask_val)
                outputs_val_soft = F.softmax(outputs_val, dim=1)  # batch_size, num_classes, H, W
                loss_dice = dice_loss_multi(outputs_val_soft, mask_val, device, is_one_hot=True)

                loss_val = loss_ce + loss_dice
                val_loss += loss_val.item()

                # calculate dice
                output_argmax = torch.argmax(outputs_val_soft, dim=1)
                dice_one = only_dice(output_argmax, mask_val)
                dice_epoch += dice_one

        val_dice_avg = dice_epoch / num_batches['val']
        logger.info(f'Epoch {epoch}/{args.num_epochs - 1} Val Loss: {val_loss / num_batches["val"]}'
                    f' Time: {time.time() - e_v_time} s')

        writer.add_scalar('Dice/val', val_dice_avg, epoch)
        writer.add_scalar('Loss/val', val_loss / num_batches['val'], epoch)

        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(path_trained_models, f'best_dice.pth'))
            logger.info(f'Best model saved at epoch {epoch}, dice: {best_dice}')

        if epoch == args.num_epochs - 1:
            final_dice = val_dice_avg

    logger.info(f'network: {args.network}, attention: {args.attention}, dataset: {dataset_name}')
    logger.info(f'Best dice: {best_dice} at epoch {best_epoch}')
    logger.info(f'final dice: {final_dice}')
    # save final model
    torch.save(model.state_dict(), os.path.join(path_trained_models, f'final.pth'))
    logger.info(f'Final model saved!')

    time_elapsed = time.time() - since
    train_final_txt = (f'Training completed in {time_elapsed // 3600:.0f}h '
                       f'{time_elapsed % 3600 // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(train_final_txt)
    writer.add_text('Training Time', train_final_txt)
    writer.close()

    # Test
    t_time = time.time()
    model.load_state_dict(torch.load(os.path.join(path_trained_models, 'best_dice.pth')))

    test_metric_dict = {
        'name': [],
        'dice': [],
        'jc': [],
        'asd': [],
        'sen': [],
        'hd95': []
    }

    png_save_path = os.path.join(path_trained_models, 'save_png')
    pred_save_path = os.path.join(path_trained_models, 'save_pred')

    for idx, data in enumerate(dataloaders['test']):
        model.eval()
        image, label = data['image'], data['label']
        image, label = image.to(device), label.to(device)
        ID = data['name'][0]
        with torch.no_grad():
            output = model(image)
            output_soft = F.softmax(output, dim=1)
            output_argmax = torch.argmax(output_soft, dim=1)

            # save png
            save_pred(output_argmax, pred_save_path, id_name=ID)
            save_img(img=image, msk=label, msk_pred=output_argmax,
                     save_path=png_save_path, id_name=ID,
                     mean=cfg['MEAN'], std=cfg['STD'], data_idx=idx)

            # calculate metric
            dice, jc, asd, sen, hd95 = calculate_metric(output_argmax, label, affine=None)
            test_metric_dict['name'].append(ID)
            test_metric_dict['dice'].append(dice)
            test_metric_dict['jc'].append(jc)
            test_metric_dict['asd'].append(asd)
            test_metric_dict['sen'].append(sen)
            test_metric_dict['hd95'].append(hd95)

            logger.info(f'idx: {idx}, ID: {ID}, dice: {dice}, jc: {jc}, asd: {asd}, sen: {sen}, hd95: {hd95}')

    test_metric_dict_df = pandas.DataFrame(test_metric_dict)
    test_metric_dict_df.to_csv(os.path.join(path_trained_models, 'test_metric.csv'))
    # calculate the mean value and std value
    mean_dice, std_dice = np.mean(test_metric_dict['dice']), np.std(test_metric_dict['dice'])
    mean_jc, std_jc = np.mean(test_metric_dict['jc']), np.std(test_metric_dict['jc'])
    mean_asd, std_asd = np.mean(test_metric_dict['asd']), np.std(test_metric_dict['asd'])
    mean_sen, std_sen = np.mean(test_metric_dict['sen']), np.std(test_metric_dict['sen'])
    mean_hd95, std_hd95 = np.mean(test_metric_dict['hd95']), np.std(test_metric_dict['hd95'])

    logger.info(f'Using best model to test')

    logger.info(f'\nDice: {mean_dice} ± {std_dice}, \n'
                f'JC: {mean_jc} ± {std_jc}, \n'
                f'ASD: {mean_asd} ± {std_asd}, \n'
                f'SEN: {mean_sen} ± {std_sen}, \n'
                f'HD95: {mean_hd95} ± {std_hd95}')

    logger.info(f'\nDice: {mean_dice:.4f} ± {std_dice:.4f}, \n'
                f'JC: {mean_jc:.4f} ± {std_jc:.4f}, \n'
                f'ASD: {mean_asd:.4f} ± {std_asd:.4f}, \n'
                f'SEN: {mean_sen:.4f} ± {std_sen:.4f}, \n'
                f'HD95: {mean_hd95:.4f} ± {std_hd95:.4f}')

    logger.info(
        f'network: {args.network}, attention: {args.attention}, dataset: {dataset_name}， test mean dice: {mean_dice:.4f}')
    logger.info(f'Test time: {time.time() - t_time} s')
