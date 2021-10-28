import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils_training.multiscale_loss import multiscaleEPE, realEPE, sparse_max_pool, multiscaleCE, FocalLoss
from matplotlib import pyplot as plt
from imageio import imread
import torchvision.transforms as tf
import os
from torchnet.meter.confusionmeter import ConfusionMeter
import torch.nn as nn
from utils_training.preprocess_batch import pre_process_data, pre_process_change, post_process_single_img_data
from utils.plot import overlay_result
from utils.evaluate import IoU

class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
        return self.loss(F.log_softmax(output, dim=1), target)

def plot_during_training(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256,
                         target_change_original,
                         target_change_256,
                         out_change_orig,
                         out_change_256,
                         mask=None, mask_256=None,
                         return_img = False):
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())
    target_change_original = 50 * target_change_original[0][0].cpu()
    out_change_original = F.interpolate(out_change_orig[1],(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    out_change_original = out_change_original[0].argmax(0)
    out_change_original = 50*out_change_original.cpu().numpy()
    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())
    target_change_256 = 50 * target_change_256[0][0].cpu().numpy()
    out_change_256 = F.interpolate(out_change_256[1],(h_256, h_256),
                                        mode='bilinear',align_corners=False)
    out_change_256 = out_change_256[0].argmax(0)
    out_change_256 = 50*out_change_256.cpu().numpy()

    fig, axis = plt.subplots(2, 7, figsize=(20, 10))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsrc image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntgtfeh image")
    if apply_mask:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsrc remapped with GT")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsrc remapped with network")
    axis[0][5].imshow(target_change_original,vmax=255)
    axis[0][5].set_title("original reso: \nGT change label")
    axis[0][6].imshow(out_change_original,vmax=255)
    axis[0][6].set_title("original reso: \nestim. change seg.")

    axis[1][0].imshow(image_1_256.numpy())
    axis[1][0].set_title("reso 256: \nsrc image")
    axis[1][1].imshow(image_2_256.numpy())
    axis[1][1].set_title("reso 256:\ntgt image")
    if apply_mask:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[1][2].set_title("reso 256: \nmask applied during training")
    axis[1][3].imshow(remapped_gt_256)
    axis[1][3].set_title("reso 256: \nsrc remapped with GT")
    axis[1][4].imshow(remapped_est_256)
    axis[1][4].set_title("reso 256: \nsrc remapped with network")
    axis[1][5].imshow(target_change_256,vmax=255)
    axis[1][5].set_title("reso 256: \nGT change label")
    axis[1][6].imshow(out_change_256,vmax=255)
    axis[1][6].set_title("reso 256: \nestim. change seg.")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    if return_img:
        vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
        return vis_result.transpose(2,0,1) # channel first

def plot_during_training2(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256,
                         target_change_original,
                         target_change_256,
                         out_change_orig,
                         out_change_256,
                         mask=None, mask_256=None,
                         return_img = False):
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())
    if target_change_original is not None:
        target_change_original = 50*target_change_original[0][0].cpu()
    out_change_original = F.interpolate(out_change_orig[1],(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    out_change_original = out_change_original[0].argmax(0)
    out_change_original = 50*out_change_original.cpu().numpy()
    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())
    target_change_256 = 50 * target_change_256[0][0].cpu().numpy()
    out_change_256 = F.interpolate(out_change_256[1],(h_256, h_256),
                                        mode='bilinear',align_corners=False)
    out_change_256 = out_change_256[0].argmax(0)
    out_change_256 = 50*out_change_256.cpu().numpy()
    num_figs=4 if target_change_original is None else 5
    fig, axis = plt.subplots(1, num_figs, figsize=(20, 10))
    axis[0].imshow(image_1.numpy())
    axis[0].set_title("src image")
    axis[1].imshow(image_2.numpy())
    axis[1].set_title("tgt image")
    if apply_mask:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[2].imshow(remapped_est)
    axis[2].set_title("src remapped with network")

    axis[3].imshow(out_change_original,vmax=255,interpolation='nearest')
    axis[3].set_title("estim. change seg.")
    if target_change_original is not None:
        axis[4].imshow(target_change_original,vmax=255,interpolation='nearest')
        axis[4].set_title("GT change label")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    if return_img:
        vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
        return vis_result.transpose(2,0,1) # channel first

def plot_during_training3(save_path, epoch, batch,
                         h_original, w_original,
                         source_image, target_image,
                         target_change_original,
                         out_change_orig,
                         return_img = False,
                         norm = 'z_score',
                         save_split=True):
    image_1,image_2 = post_process_single_img_data(source_image[0],target_image[0],norm=norm)

    if target_change_original is not None:
        target_change_original = 50*target_change_original[0].cpu().squeeze()
    out_change_original = F.interpolate(out_change_orig,(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    out_change_original = out_change_original[0].argmax(0)
    out_change_original = 50*out_change_original.cpu().numpy()


    if save_split:
        if not os.path.isdir(os.path.join(save_path,'pred_on_t1')): os.mkdir(os.path.join(save_path,'pred_on_t1'))
        if not os.path.isdir(os.path.join(save_path,'gt_on_t1')): os.mkdir(os.path.join(save_path,'gt_on_t1'))
        out_change = overlay_result(out_change_original[:,:,None].astype(np.bool8),image_2.numpy())
        target_change_original = overlay_result(target_change_original[:,:,None].bool().numpy(),image_2.numpy())
        plt.imsave('{}/pred_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change)
        plt.imsave('{}/gt_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),target_change_original)
        if return_img:
            return target_change_original.transpose(2,0,1)
    else:
        num_figs=4
        fig, axis = plt.subplots(1, num_figs, figsize=(20, 10))
        axis[0].imshow(image_1.numpy())
        axis[0].set_title("src image")
        axis[1].imshow(image_2.numpy())
        axis[1].set_title("tgt image")

        axis[2].imshow(out_change_original,vmax=255,interpolation='nearest')
        axis[2].set_title("estim. change seg.")
        if target_change_original is not None:
            axis[3].imshow(target_change_original,vmax=255,interpolation='nearest')
            axis[3].set_title("GT change label")
        fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                    bbox_inches='tight')
        plt.close(fig)
        if return_img:
            vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
            return vis_result.transpose(2,0,1) # channel first


def plot_infer_only(save_path, epoch, batch,
                         h_original, w_original,
                         source_image, target_image,
                         target_change_original,
                         out_change_orig,
                         return_img = False,
                         norm = 'z_score'):
    image_1,image_2 = post_process_single_img_data(source_image[0],target_image[0],norm=norm)
    # # resolution original
    # mean_values = torch.tensor([0.485, 0.456, 0.406],
    #                            dtype=source_image.dtype).view(3, 1, 1)
    # std_values = torch.tensor([0.229, 0.224, 0.225],
    #                           dtype=source_image.dtype).view(3, 1, 1)
    # image_1 = (source_image.detach()[0].cpu() * std_values +
    #            mean_values).clamp(0, 1).permute(1, 2, 0)
    # image_2 = (target_image.detach()[0].cpu() * std_values +
    #            mean_values).clamp(0, 1).permute(1, 2, 0)

    if target_change_original is not None:
        target_change_original = 50*target_change_original[0].cpu().squeeze()
    out_change_original = F.interpolate(out_change_orig,(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    out_change_original = out_change_original[0].argmax(0)
    out_change_original = 50*out_change_original.cpu().numpy()

    num_figs=1
    fig, axis = plt.subplots(1, num_figs, figsize=(4, 10))
    # axis[0].imshow(image_1.numpy())
    # axis[0].set_title("src image")
    # axis[1].imshow(image_2.numpy())
    # axis[1].set_title("tgt image")

    axis.imshow(out_change_original,vmax=255,interpolation='nearest')
    axis.set_title("estim. change seg.")
    # if target_change_original is not None:
    #     axis[3].imshow(target_change_original,vmax=255,interpolation='nearest')
    #     axis[3].set_title("GT change label")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    if return_img:
        vis_result = imread('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch)).astype(np.uint8)[:,:,:3]
        return vis_result.transpose(2,0,1) # channel first




def train_epoch(args, net,
                optimizer,
                train_loader,
                device,
                epoch,
                writer,
                save_path,

):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        writer: for tensorboard
    Output:
        running_total_loss: total training loss

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.
    """
    n_iter = epoch*len(train_loader)
    net.train()
    running_total_loss = 0
    running_flow_loss = 0
    running_change_loss = 0
    weight = torch.ones(2)
    criterion = criterion_CEloss(weight.cuda())

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        # pre-process the data
        source_image, target_image, source_image_256, target_image_256 = \
            pre_process_data(mini_batch['source_image'],
            mini_batch['target_image'],
            device=device, norm=args.img_norm_type,rgb_order=args.rgb_order)
        source_change, target_change, source_change_256, target_change_256 = \
            pre_process_change(mini_batch['source_change'],
            mini_batch['target_change'],
            device=device)
        disable_flow = mini_batch['disable_flow'][..., None, None].to(device)  # bs,1,1,1
        out_dict = net(target_image, source_image, target_image_256, source_image_256,disable_flow=disable_flow)
        # out_flow_256, out_flow_orig = out_dict['flow']
        out_change_orig = out_dict['change']
        use_flow = mini_batch['use_flow'][...,None].to(device)


        Loss_total = criterion(out_change_orig,target_change.squeeze())

        Loss_total.backward()
        optimizer.step()

        running_total_loss += Loss_total.item()


        writer.add_scalar('train_total_per_iter', Loss_total.item(), n_iter)

        n_iter += 1
        pbar.set_description(
                'training: R_change_loss: %.3f/%.3f' % (running_change_loss / (i + 1),
                                             Loss_total.item()))
    running_total_loss /= len(train_loader)


    return dict(total=running_total_loss,change=running_change_loss,flow=running_flow_loss,
                # accuracy = Acc,
                # IoUs=IoUs,
                # mIoU = mIoU,
                # f1=f1
                )


def validate_epoch(args, net,
                   val_loader,
                   device,
                   epoch,
                   save_path,
                   writer,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False,
                   sparse=False,
                   robust_L1_loss=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
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
    n_iter = epoch*len(val_loader)
    confmeter = ConfusionMeter(k=net.module.num_class,normalized=False)

    net.eval()
    if loss_grid_weights is None:
        loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]
    change_criterion = FocalLoss()
    running_total_loss = 0
    if not os.path.isdir(save_path): os.mkdir(save_path)

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)
        CE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)

        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = \
                pre_process_data(mini_batch['source_image'],
                                 mini_batch['target_image'],
                                 device=device, norm=args.img_norm_type, rgb_order=args.rgb_order)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(mini_batch['source_change'],
                                   mini_batch['target_change'],
                                   device=device)
            disable_flow = mini_batch['disable_flow'][..., None,None].to(device) # bs,1,1,1
            out_dict = net(target_image, source_image, target_image_256, source_image_256, disable_flow=disable_flow)
            out_change_orig = out_dict['change']
            # ''' Evaluate Change '''
            bs, _, h_original, w_original = target_image.shape

            out_change_orig = torch.nn.functional.interpolate(out_change_orig.detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            out_change_orig = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1)
            confmeter.add(out_change_orig, target_change.squeeze().long())

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU


    return dict(accuracy = Acc,
                IoUs=IoUs, mIoU = mIoU, f1=f1)



def test_epoch(args, net,
               test_loader,
               device,
               epoch,
               save_path,
               writer,
               div_flow=1,
               plot_interval=10
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
    confmeter = ConfusionMeter(k=net.module.num_class,normalized=False)
    weight = torch.ones(2)

    criterion = criterion_CEloss(weight.cuda())

    net.eval()

    if not os.path.isdir(save_path): os.mkdir(save_path)
    print('Begin Testing {}'.format(save_path))
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = \
                pre_process_data(mini_batch['source_image'],
                                 mini_batch['target_image'],
                                 device=device, norm=args.img_norm_type, rgb_order=args.rgb_order)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(mini_batch['source_change'],
                                   mini_batch['target_change'],
                                   device=device)
            disable_flow = mini_batch['disable_flow'][..., None,None].to(device) # bs,1,1,1
            out_dict = net(target_image, source_image, target_image_256, source_image_256, disable_flow=disable_flow)

            out_change_orig = out_dict['change']

            bs, _, h_original, w_original = source_image.shape
            bs, _, h_256, w_256 = source_image_256.shape

            if i % plot_interval == 0:
                vis_img = plot_during_training3(save_path, epoch, i,
                                               h_original, w_original,
                                               source_image, target_image,
                                               target_change_original=target_change,
                                               out_change_orig=out_change_orig,
                                               return_img=True,
                                               norm = args.img_norm_type)
                writer.add_image('val_warping_per_iter', vis_img, n_iter)

            out_change_orig = torch.nn.functional.interpolate(out_change_orig.detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            out_change_orig = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1)
            confmeter.add(out_change_orig, target_change.squeeze().long())

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU

    return dict(accuracy = Acc, IoUs=IoUs, mIoU = mIoU, f1=f1)

