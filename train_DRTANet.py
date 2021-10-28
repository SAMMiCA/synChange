import numpy as np
import argparse
import time
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from datasets.training_dataset import HomoAffTps_Dataset
from datasets.load_pre_made_dataset import PreMadeChangeDataset
from utils_training.optimize_DRTANet import train_epoch, validate_epoch, test_epoch
from models.DR_TANet.TANet import TANet
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from tensorboardX import SummaryWriter
from utils.image_transforms import ArrayToTensor
from datasets.vl_cmu_cd import vl_cmu_cd_eval
from datasets.pcd import gsv_eval, tsunami_eval,pcd_5fold
from datasets.changesim import changesim_eval
from datasets.prepare_dataloaders import prepare_trainval,prepare_test
from datasets.prepare_transforms import prepare_transforms
from utils_training.prepare_optimizer import prepare_optim
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='GLU-Net train script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--training_data_dir', type=str, default='../dataset/train_datasets',
                        help='path to directory containing original images for training')
    parser.add_argument('--evaluation_data_dir', type=str,  default='../dataset/test_datasets',
                        help='path to directory containing original images for validation')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained',
                        # default='pre_trained_models/GLUNet_DPED_CityScape_ADE.pth',
                        help='path to pre-trained model (load only model params)')
    parser.add_argument('--resume', dest='resume',
                       # default='snapshots/2021_10_03_09_27/epoch_8.pth',
                       help='path to resume model (load both model and optimizer params')
    parser.add_argument('--multi_class', action='store_true',
                        help='if true, do multi-class change detection')
    parser.add_argument('--trainset_list', nargs='+', default=['pcd'])
    parser.add_argument('--testset_list', nargs='+', default=['pcd'])
    parser.add_argument('--valset_list', nargs='+', default=['synthetic'])

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, # for RTX3090
                        help='train/val batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, # for RTX3090
                        help='test batch size')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    parser.add_argument('--split_ratio', type=float, default=0.99,
                        help='train/val split ratio')
    parser.add_argument('--split2_ratio', type=float, default=0.01,
                        help='val/not-used split ratio (if 0.9, use 90% of val samples)')
    parser.add_argument('--plot_interval', type=int, default=1,
                        help='plot every N iteration in test_epoch')
    parser.add_argument('--test_interval', type=int, default=10,
                        help='test every N epoch')
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[12,20,23], # for 25 epoch
                        help='schedule for learning rate decrease when multi-stepLR')
    parser.add_argument('--optim', type=str, default='adam',
                        help='adam or adamw')
    parser.add_argument('--scheduler', type=str, default='lambda',
                        help='lambda or multistep')
    parser.add_argument('--train_img_size', nargs='+', type=int,
                        default=[256,256],
                        help='img_size (if you want to use synthetic dataset, this value should be (520,520)')
    parser.add_argument('--test_img_size', nargs='+', type=int,
                        default=[256,256],
                        help='img_size (if you want to use synthetic dataset, this value should be (520,520)')
    parser.add_argument('--disable_transform', action='store_false',
                        help='if true, do not perform transform when training')
    parser.add_argument('--img_norm_type',type=str, default='min_max',
                        help='z_score or min_max')
    parser.add_argument('--rgb_order', type=str, default='bgr',
                        help='rgb or bgr')
    parser.add_argument('--test_only', action='store_true',
                        help='if true, do test only')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:{}'.format(device))
    print('-----------------------Arguments-----------------------------')
    for arg in vars(args):
        print('{}:{}'.format(arg, getattr(args, arg)))
    print('-------------------------------------------------------------')

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # transforms
    source_img_transforms, target_img_transforms,co_transform, flow_transform, change_transform = prepare_transforms(args)

    # dataloaders
    train_dataloader, val_dataloader = prepare_trainval(args, source_img_transforms,
                                                        target_img_transforms,
                                                        flow_transform, co_transform, change_transform)
    test_dataloaders = prepare_test(args, source_img_transforms=transforms.Compose([ArrayToTensor(get_float=False)]),
                                    target_img_transforms=transforms.Compose([ArrayToTensor(get_float=False)]),
                                    flow_transform=flow_transform, co_transform=None, change_transform=change_transform)

    # models
    model = TANet()
    print(colored('==> ', 'blue') + 'DR-TANet created.')

    # Optimizer
    optimizer, scheduler = prepare_optim(args,model)


    if args.pretrained:
        # reload from pre_trained_model
        model, _, _, _, _ = load_checkpoint(model, None, None, filename=args.pretrained)
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = float("inf")
        start_epoch = 0

    elif args.resume:
        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                 filename=args.resume)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.resume))
    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = float("inf")
        start_epoch = 0

    # confusionmeter
    num_class = 5 if args.multi_class else 2

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'test'))

    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()
    if not args.test_only:
        for epoch in range(start_epoch, args.n_epoch):
            print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_last_lr()[0]))

            train_loss = train_epoch(args, model,
                                     optimizer,
                                     train_dataloader,
                                     device,
                                     epoch,
                                     train_writer,
                                     save_path=os.path.join(save_path, 'train'),
                                     )
            scheduler.step()
            train_writer.add_scalar('train loss: change(FE)', train_loss['change'], epoch)
            train_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
            print(colored('==> ', 'green') + 'Train average loss:', train_loss['total'])

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.module.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(),
                             'best_loss': 9999999},
                            False, save_path, 'epoch_{}.pth'.format(epoch + 1))

            if epoch % args.test_interval == 0:
                for dataset_name,test_dataloader in test_dataloaders.items():
                    result = test_epoch(args, model, test_dataloader, device, epoch=epoch,
                               save_path=os.path.join(save_path, dataset_name),
                               writer=val_writer,
                               plot_interval=args.plot_interval)
                    print('          F1: {:.2f}, Accuracy: {:.2f} '.format(result['f1'], result['accuracy']))
                    print('          Static  |   Change   |   mIoU ')
                    print('          %7.2f %7.2f %7.2f ' %
                          (result['IoUs'][0], result['IoUs'][-1], result['mIoU']))



                # Validation
                # result = \
                #     validate_epoch(args, model, val_dataloader, device, epoch=epoch,
                #                    save_path=os.path.join(save_path, 'val'),
                #                    writer = val_writer,
                #                    )
                #
                # print('          F1: {:.2f}, Accuracy: {:.2f} '.format(result['f1'], result['accuracy']))
                # print('          Static  |   Change   |   mIoU ')
                # print('          %7.2f %7.2f %7.2f ' %
                #       (result['IoUs'][0], result['IoUs'][-1], result['mIoU']))
                # print(colored('==> ', 'blue') + 'finished epoch :', epoch + 1)
                #
                # # save checkpoint for each epoch and a fine called best_model so far
                # is_best = result['f1'] < best_val
                # best_val = min(result['f1'], best_val)
                # save_checkpoint({'epoch': epoch + 1,
                #                  'state_dict': model.module.state_dict(),
                #                  'optimizer': optimizer.state_dict(),
                #                  'scheduler': scheduler.state_dict(),
                #                  'best_loss': best_val},
                #                 is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

        print(args.seed, 'Training took:', time.time()-train_started, 'seconds')

    else:
        for dataset_name, test_dataloader in test_dataloaders.items():
            result = test_epoch(args, model, test_dataloader, device, epoch=start_epoch,
                                save_path=os.path.join(save_path, dataset_name),
                                writer=val_writer,
                                plot_interval=args.plot_interval)
            print('          F1: {:.2f}, Accuracy: {:.2f} '.format(result['f1'], result['accuracy']))
            print('          Static  |   Change   |   mIoU ')
            print('          %7.2f %7.2f %7.2f ' %
                  (result['IoUs'][0], result['IoUs'][-1], result['mIoU']))
