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
from datasets.pcd import gsv_eval, tsunami_eval
from datasets.changesim import changesim_eval
from datasets.desk import desk_demo
from datasets.ai2thor import ai2thor
import albumentations as A
from tqdm import tqdm
import torch.nn.functional as F
from utils_training.optimize_GLUChangeNet import pre_process_change,pre_process_data,plot_during_training2



def viz_epoch(net,
               test_loader,
               device,
               epoch,
               save_path,
               writer,
               div_flow=1
               ):
    """
    Test epoch script
    Args:
        net: model architecture
        test_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """
    n_iter = epoch*len(test_loader)

    net.eval()

    if not os.path.isdir(save_path): os.mkdir(save_path)

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                mini_batch['source_image'],
                mini_batch['target_image'],
                device=device)
            if 'source_change' in mini_batch.keys():
                source_change, target_change, source_change_256, target_change_256 = \
                    pre_process_change(mini_batch['source_change'],
                                       mini_batch['target_change'],
                                       device=device)
            else:
                source_change, target_change, source_change_256, target_change_256 = \
                    pre_process_change(torch.zeros_like(mini_batch['source_image'][:,0]),
                                       torch.zeros_like(mini_batch['target_image'][:,0]),
                                       device=device)
            out_dict = net(target_image, source_image, target_image_256, source_image_256)
            out_flow_256, out_flow_orig = out_dict['flow']
            out_change_256, out_change_orig = out_dict['change']
            bs, _, h_original, w_original = source_image.shape
            bs, _, h_256, w_256 = source_image_256.shape

            flow_gt_original = F.interpolate(out_flow_orig[-1], (h_original, w_original),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            flow_gt_256 = F.interpolate(out_flow_256[-1], (h_256, w_256),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            if i % 1 == 0:
                vis_img = plot_during_training2(save_path, epoch, i, False,
                                               h_original, w_original, h_256, w_256,
                                               source_image, target_image, source_image_256, target_image_256, div_flow,
                                               flow_gt_original, flow_gt_256, output_net=out_flow_orig[-1],
                                               output_net_256=out_flow_256[-1],
                                               target_change_original=target_change,
                                               target_change_256=target_change_256,
                                               out_change_orig=out_change_orig,
                                               out_change_256=out_change_256,
                                               return_img=False)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='GLU-Net train script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--pre_loaded_training_dataset', default=True, type=boolean_string,
                        help='Synthetic training dataset is already created and saved in disk ? default is False')
    parser.add_argument('--training_data_dir', type=str, default='result3',
                        help='path to directory containing original images for training if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of training images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--evaluation_data_dir', type=str,  default='/media/rit/GLU-CHANGE-SSD500/dataset/',
                        help='path to directory containing original images for validation if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of validation images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained',
                        # default='pre_trained_models/GLUNet_DPED_CityScape_ADE.pth',
                        help='path to pre-trained model (load only model params)')
    parser.add_argument('--resume', dest='resume',
                       default='snapshots/2021_09_23_19_17/epoch_15.pth',
                       help='path to resume model (load both model and optimizer params')
    parser.add_argument('--multi_class', action='store_true',
                        help='if true, do multi-class change detection')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=4e-4,
                        help='weight decay constant')
    parser.add_argument('--div_flow', type=float, default=1.0,
                        help='div flow')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # datasets, pre-processing of the images is done within the network function !
    co_transform = A.Compose([
        A.RandomCrop(height=224,width=320),
        A.Resize(height=480,width=640)
    ])
    # If synthetic pairs were already created and saved to disk, run instead of 'train_dataset' the following.
    # and replace args.training_data_dir by the root to folders containing images/ and flow/

    flow_transform = transforms.Compose([ArrayToTensor()]) # just put channels first and put it to float
    change_transform = transforms.Compose([ArrayToTensor()])

    test_datasets = {}
    test_datasets['ai2thor'] = ai2thor(root=os.path.join(args.evaluation_data_dir,'ai2thor'),
                                  source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
                                  target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
                                  )
    test_datasets['desk2'] = desk_demo(root=os.path.join(args.evaluation_data_dir,'desk'),
                                  source_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
                                  target_image_transform=transforms.Compose([ArrayToTensor(get_float=False)]),
                                  )

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
    #                               )

    # Dataloader

    test_dataloaders = {k:DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)
                        for k, test_dataset in test_datasets.items()}

    # models
    model = GLUChangeNet_model(batch_norm=True, pyramid_type='VGG',
                                 div=args.div_flow, evaluation=False,
                                 consensus_network=False,
                                 cyclic_consistency=True,
                                 dense_connection=True,
                                 decoder_inputs='corr_flow_feat',
                                 refinement_at_all_levels=False,
                                 refinement_at_adaptive_reso=True)
    print(colored('==> ', 'blue') + 'GLU-Change-Net created.')

    # Optimizer
    optimizer = \
        optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[65, 75, 95],
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

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'test'))

    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()
    for dataset_name, test_dataloader in test_dataloaders.items():
        #try:
            viz_epoch(model, test_dataloader, device, epoch=start_epoch,
                       save_path=os.path.join(save_path, dataset_name),
                       writer=val_writer,
                       div_flow=args.div_flow)
        #except:pass