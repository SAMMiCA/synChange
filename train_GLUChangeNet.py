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
from utils_training.optimize_GLUChangeNet import train_epoch, validate_epoch, test_epoch, train_change
from models.our_models.GLUChangeNet import GLUChangeNet_model
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from tensorboardX import SummaryWriter
from utils.image_transforms import ArrayToTensor
from datasets.vl_cmu_cd import vl_cmu_cd_eval
from datasets.pcd import gsv_eval, tsunami_eval,pcd_5fold
from datasets.changesim import changesim_eval
from datasets.prepare_dataloaders import prepare_trainval,prepare_test
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
                        default='pre_trained_models/GLUNet_DPED_CityScape_ADE.pth',
                        help='path to pre-trained model (load only model params)')
    parser.add_argument('--resume', dest='resume',
                       # default='snapshots/2021_10_03_09_27/epoch_8.pth',
                       help='path to resume model (load both model and optimizer params')
    parser.add_argument('--multi_class', action='store_true',
                        help='if true, do multi-class change detection')
    parser.add_argument('--trainset_list', nargs='+')
    parser.add_argument('--testset_list', nargs='+')
    parser.add_argument('--valset_list', nargs='+', default=['synthetic'])


    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=24, # for RTX3090
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=4,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=4e-4,
                        help='weight decay constant')
    parser.add_argument('--div_flow', type=float, default=1.0,
                        help='div flow')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    parser.add_argument('--split_ratio', type=float, default=0.99,
                        help='train/val split ratio')
    parser.add_argument('--split2_ratio', type=float, default=0.99,
                        help='val/not-used split ratio (if 0.9, use 90% of val samples)')
    parser.add_argument('--plot_interval', type=int, default=10,
                        help='plot every N iteration')
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[12,20,23], # for 25 epoch
                        help='schedule for learning rate decrease')

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
    source_img_transforms = A.Compose([
        A.ColorJitter(p=0.5),
        A.RandomShadow(shadow_roi=(0,0,1,1), p=0.5),
        A.OneOf([A.ChannelDropout(p=0.5),
                 A.ChannelShuffle(p=0.5),
                 A.ToGray(p=0.5),
                 A.ToSepia(p=0.5)]),
        A.OneOf([A.RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.5,p=0.5),
                A.RandomRain(p=0.5),
                 A.RandomSnow(p=0.5),
                 A.RandomSunFlare(src_radius=150,p=0.5)]),
        A.OneOf([A.Compose([A.CropAndPad(percent=-0.07), A.CropAndPad(percent=0.07)]),
                 A.Compose([A.CropAndPad(percent=-0.03), A.CropAndPad(percent=0.03)]),
                 A.Compose([]),
                 ]),
        ToTensorV2()])
    target_img_transforms = A.Compose([
        A.ColorJitter(p=0.5),
        A.RandomShadow(shadow_roi=(0,0,1,1), p=0.5),
        A.OneOf([A.ChannelDropout(p=0.5),
                 A.ChannelShuffle(p=0.5),
                 A.ToGray(p=0.5),
                 A.ToSepia(p=0.5)]),
        A.OneOf([A.RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.7,p=0.5),
                A.RandomRain(p=0.5),
                 A.RandomSnow(p=0.5),
                 A.RandomSunFlare(src_radius=150,p=0.5)]),
        A.OneOf([A.Compose([A.CropAndPad(percent=-0.07), A.CropAndPad(percent=0.07)]),
                 A.Compose([A.CropAndPad(percent=-0.03), A.CropAndPad(percent=0.03)]),
                 A.Compose([]),
                 ]),
        ToTensorV2()])
    co_transform = None
    # for panoramic images
    # co_transform = A.Compose([
    #     A.RandomCrop(height=224,width=320),
    #     A.Resize(height=480,width=640)
    # ])
    flow_transform = transforms.Compose([ArrayToTensor()]) # just put channels first and put it to float
    change_transform = transforms.Compose([ArrayToTensor()])

    # dataloaders
    train_dataloader, val_dataloader = prepare_trainval(args, source_img_transforms, target_img_transforms,
                     flow_transform, co_transform, change_transform)
    test_dataloaders = prepare_test(args, source_img_transforms=transforms.Compose([ArrayToTensor(get_float=False)]),
                                    target_img_transforms=transforms.Compose([ArrayToTensor(get_float=False)]),
                     flow_transform=flow_transform, co_transform=None, change_transform=change_transform)

    # train_datasets = {}
    # train_datasets['synthetic'], val_dataset = PreMadeChangeDataset(root=args.training_data_dir,
    #                                   source_image_transform=source_img_transforms,
    #                                   target_image_transform=target_img_transforms,
    #                                   flow_transform=flow_transform,
    #                                   co_transform=None,
    #                                   change_transform=change_transform,
    #                                   split=args.split_ratio,
    #                                   split2=args.split2_ratio,
    #                                   multi_class =args.multi_class)  # train:val = 95:5
    # train_datasets['vl_cmu_cd'] =vl_cmu_cd_eval(root=os.path.join(args.evaluation_data_dir,'VL-CMU-CD'),
    #                               source_image_transform=source_img_transforms,
    #                               target_image_transform=target_img_transforms,
    #                               change_transform=change_transform,
    #                               split= 'train',
    #                               img_size = (520,520)
    #                               )
    # # train_datasets['pcd'] =pcd_5fold(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
    # #                               source_image_transform=source_img_transforms,
    # #                               target_image_transform=target_img_transforms,
    # #                               change_transform=change_transform,
    # #                               split= 'train',
    # #                               img_size = (520,520)
    # #                               )
    # for k, d in train_datasets.items():
    #     print('LOADING train split of {} ({} pairs)'.format(k,len(d)))
    #
    # train_dataset = torch.utils.data.ConcatDataset([ d for k,d in train_datasets.items()])
    # print('# of training samples in total: ({} pairs)'.format(len(train_dataset)))

    # test_datasets = {}

    # test_datasets['changesim_normal'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'ChangeSim'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               multi_class=False,
    #                               split='Seq_0'
    #                               )
    # test_datasets['changesim_dark'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'ChangeSim'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               multi_class=False,
    #                               split='Seq_0_dark'
    #                               )
    # test_datasets['changesim_dust'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'ChangeSim'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               multi_class=False,
    #                               split='Seq_0_dust'
    #                               )
    # test_datasets['changesim_storage'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'ChangeSim'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               multi_class=False,
    #                               mapname='Storage',
    #                                split='Seq_0'
    #                               )
    # test_datasets['changesim_fire'] = changesim_eval(root=os.path.join(args.evaluation_data_dir,'ChangeSim'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               multi_class=False,
    #                               split='Seq_0_fire'
    #                               )
    # test_datasets['gsv'] = gsv_eval(root=os.path.join(args.evaluation_data_dir,'GSV'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               co_transform=co_transform,
    #                               change_transform=change_transform,
    #                               )
    #
    # test_datasets['tsunami'] = tsunami_eval(root=os.path.join(args.evaluation_data_dir,'TSUNAMI'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               co_transform=co_transform,
    #                               change_transform=change_transform,
    #                               )
    # test_datasets['vl_cmu_cd'] = vl_cmu_cd_eval(root=os.path.join(args.evaluation_data_dir,'VL-CMU-CD'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               split='test'
    #                               )
    # test_datasets['pcd'] = vl_cmu_cd_eval(root=os.path.join(args.evaluation_data_dir,'pcd_5cv'),
    #                               source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
    #                               change_transform=change_transform,
    #                               split='test'
    #                               )
    # for k, d in test_datasets.items():
    #     print('LOADING test split of {} ({} pairs)'.format(k,len(d)))

    # Dataloader
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.n_threads,
    #                               pin_memory=True)
    # val_dataloader = DataLoader(val_dataset,
    #                             batch_size=args.batch_size,
    #                             shuffle=True,
    #                             num_workers=args.n_threads)

    # test_dataloaders = {k:DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=args.n_threads)
    #                     for k, test_dataset in test_datasets.items()}

    # models
    model = GLUChangeNet_model(batch_norm=True, pyramid_type='VGG',
                                 div=args.div_flow, evaluation=False,
                                 consensus_network=False,
                                 cyclic_consistency=True,
                                 dense_connection=True,
                                 decoder_inputs='corr_flow_feat',
                                 refinement_at_all_levels=False,
                                 refinement_at_adaptive_reso=True,
                                 num_class=5 if args.multi_class else 2)
    print(colored('==> ', 'blue') + 'GLU-Change-Net created.')

    # Optimizer
    optimizer = \
        optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=args.milestones,
                                         gamma=0.5)
    weights_loss_coeffs = [0.32, 0.08, 0.02, 0.01]

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

    for epoch in range(start_epoch, args.n_epoch):
        print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_last_lr()[0]))

        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                 div_flow=args.div_flow,
                                 save_path=os.path.join(save_path, 'train'),
                                 loss_grid_weights=weights_loss_coeffs)
        scheduler.step()
        train_writer.add_scalar('train loss: flow(EPE)', train_loss['flow'], epoch)
        train_writer.add_scalar('train loss: change(FE)', train_loss['change'], epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss['total'])

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': 9999999},
                        False, save_path, 'epoch_{}.pth'.format(epoch + 1))

        for dataset_name,test_dataloader in test_dataloaders.items():
            result = test_epoch(model, test_dataloader, device, epoch=epoch,
                       save_path=os.path.join(save_path, dataset_name),
                       writer=val_writer,
                       div_flow=args.div_flow,
                       plot_interval=args.plot_interval)
            print('          F1: {:.2f}, Accuracy: {:.2f} '.format(result['f1'], result['accuracy']))
            print('          Static  |   Change   |   mIoU ')
            print('          %7.2f %7.2f %7.2f ' %
                  (result['IoUs'][0], result['IoUs'][-1], result['mIoU']))



        # Validation
        result = \
            validate_epoch(model, val_dataloader, device, epoch=epoch,
                           save_path=os.path.join(save_path, 'val'),
                           writer = val_writer,
                           div_flow=args.div_flow,
                           loss_grid_weights=weights_loss_coeffs)
        val_loss_grid, val_mean_epe, val_mean_epe_H_8, val_mean_epe_32, val_mean_epe_16  = \
            result['total'],result['mEPEs'][0].item(), result['mEPEs'][1].item(), result['mEPEs'][2].item(), result['mEPEs'][3].item()

        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('mean EPE is {}'.format(val_mean_epe))
        print('mean EPE from reso H/8 is {}'.format(val_mean_epe_H_8))
        print('mean EPE from reso 32 is {}'.format(val_mean_epe_32))
        print('mean EPE from reso 16 is {}'.format(val_mean_epe_16))
        val_writer.add_scalar('validation images: mean EPE ', val_mean_epe, epoch)
        val_writer.add_scalar('validation images: mean EPE_from_reso_H_8', val_mean_epe_H_8, epoch)
        val_writer.add_scalar('validation images: mean EPE_from_reso_32', val_mean_epe_32, epoch)
        val_writer.add_scalar('validation images: mean EPE_from_reso_16', val_mean_epe_16, epoch)
        val_writer.add_scalar('validation images: val loss', val_loss_grid, epoch)

        print('          F1: {:.2f}, Accuracy: {:.2f} '.format(result['f1'], result['accuracy']))
        print('          Static  |   Change   |   mIoU ')
        print('          %7.2f %7.2f %7.2f ' %
              (result['IoUs'][0], result['IoUs'][-1], result['mIoU']))
        print(colored('==> ', 'blue') + 'finished epoch :', epoch + 1)

        # save checkpoint for each epoch and a fine called best_model so far
        is_best = result['f1'] < best_val
        best_val = min(result['f1'], best_val)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
