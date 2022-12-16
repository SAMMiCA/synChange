import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils_training.multiscale_loss import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from imageio import imread
import torchvision.transforms as tf
import os
from torchnet.meter.confusionmeter import ConfusionMeter
from utils_training.preprocess_batch import pre_process_change,pre_process_data
from utils.plot import overlay_result
from utils.evaluate import IoU
from pytorch_msssim import ms_ssim, ssim
import flow_vis
import cv2
from models.our_models.mod import warp
from datasets.changesim import SegHelper

def calc_flow_std(flow, patch_size=16, patch_stride=16):
    # flow: B 2 H W
    flow_patches = flow.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
    b, c, num_patch_h, num_patch_w, patch_h, patch_w = flow_patches.shape
    flow_patches = flow_patches.reshape(b, c, num_patch_h, num_patch_w, patch_h * patch_w)
    std_map = flow_patches.std(dim=4).mean(dim=1)
    flow_patches = flow_patches.reshape(b, c, num_patch_h * num_patch_w, patch_h * patch_w)

    flow_stds = flow_patches.std(dim=3).mean(dim=2).mean(dim=1)
    return flow_stds, std_map


def fused_feat_sim(feats_src, feats_tgt, flow, h_img, hw=None):
    num_feats = len(feats_src)
    feat_sim_list = []
    h_f, w_f = flow.size()[-2], flow.size()[-1] if hw is None else hw
    vmask = warp(None, flow*(h_f / h_img))
    for f_src, f_tgt in zip(feats_src, feats_tgt):
        _, _, h, w = f_src.size()
        div_factor = h / h_img  # for scale flow1 value
        flow_ds = F.interpolate(flow, size=(h, w), mode='bilinear')
        warped_f_src = warp(f_src, flow_ds*div_factor)
        cos_sim = F.cosine_similarity(warped_f_src, f_tgt, dim=1).unsqueeze(1)
        # non-negative similarity
        # negative correlations are treated as orthogonal
        feat_sim = F.interpolate(cos_sim, size=(h_f, w_f), mode='bilinear')
        feat_sim_list.append(F.relu(feat_sim))
        # feat_sim_list.append((1.0+feat_sim)/2.0)
    
    fused_feat_sim = (torch.prod(torch.stack(feat_sim_list), dim=0)+1e-12)
        
    return fused_feat_sim, vmask.unsqueeze(1)

def fused_feat_diff(feats_src, feats_tgt, flow, h_img, hw=None):
    num_feats = len(feats_src)
    feat_diff_list = []
    h_f, w_f = flow.size()[-2], flow.size()[-1] if hw is None else hw
    vmask = warp(None, flow*(h_f / h_img))
    for f_src, f_tgt in zip(feats_src, feats_tgt):
        _, _, h, w = f_src.size()
        div_factor = h / h_img  # for scale flow1 value
        flow_ds = F.interpolate(flow, size=(h, w), mode='bilinear')
        warped_f_src = warp(f_src, flow_ds*div_factor)
        l1_dist = (warped_f_src - f_tgt).abs().sum(dim=1, keepdim=True)
        # non-negative similarity
        # negative correlations are treated as orthogonal
        feat_diff = F.interpolate(l1_dist, size=(h_f, w_f), mode='bilinear')
        # feat_sim_list.append(F.relu(feat_sim))
        feat_diff_list.append(feat_diff)
    
    fused_feat_diff = (torch.prod(torch.stack(feat_diff_list), dim=0))
        
    return fused_feat_diff, vmask.unsqueeze(1)


def plot_during_training2(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256,
                         target_change_original,
                         target_change_256,
                         out_change_orig,
                         out_change_256,
                         mask=None, mask_256=None,
                         f_sim_map=None, vmask=None,
                         return_img = False,
                         save_split=False,
                         seg_helper = SegHelper(),
                         multi_class=1):
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
    
    # remapped_gt = remap_using_flow_fields(image_1.numpy(),
    #                                       flow_target_x.cpu().numpy(),
    #                                       flow_target_y.cpu().numpy())
    
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())
    if target_change_original is not None:
        target_change_original = 50*target_change_original[0][0].cpu()
    out_change_original = F.interpolate(out_change_orig[-1],(h_original, w_original),
                                        mode='bilinear',align_corners=False)
    vmask = F.interpolate(vmask.float(), size=(h_original, w_original), mode='bilinear', align_corners=False).bool()
    if multi_class > 1:
        out_change_original = out_change_original[0].argmax(0)
    else:
        out_change_original = ((torch.sigmoid(out_change_original[0])).le(0.5).long()*vmask[0]).squeeze(0)
    out_change_original = 50*out_change_original.cpu().numpy()
    
    f_sim_map = F.interpolate(f_sim_map, size=(h_original, w_original),
                              mode='bilinear', align_corners=False).permute(0, 2, 3, 1)[0].cpu().numpy()
    # resolution 256x256
    # flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
    #                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
    
    # # for batch 0
    # flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    # flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    # flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    # flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    # assert flow_est_x_256.shape == flow_target_x_256.shape

    # image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
    #                mean_values).clamp(0, 1).permute(1, 2, 0)
    
    # # image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
    # #                mean_values).clamp(0, 1).permute(1, 2, 0)
    # # remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
    # #                                           flow_target_x_256.cpu().numpy(),
    # #                                           flow_target_y_256.cpu().numpy())
    # # remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
    # #                                            flow_est_y_256.cpu().numpy())
    
    # target_change_256 = 50 * target_change_256[0][0].cpu().numpy()
    # out_change_256 = F.interpolate(out_change_256[1],(h_256, h_256),
    #                                     mode='bilinear',align_corners=False)
    # if multi_class > 1:
    #     out_change_256 = out_change_256[0].argmax(0)
    # else:
    #     out_change_256 = (torch.sigmoid(out_change_256[0])).le(0.5).long().squeeze(0)
    # out_change_256 = 50*out_change_256.cpu().numpy()


    if save_split:
        if not os.path.isdir(os.path.join(save_path,'t0')): os.mkdir(os.path.join(save_path,'t0'))
        if not os.path.isdir(os.path.join(save_path,'t1')): os.mkdir(os.path.join(save_path,'t1'))
        if not os.path.isdir(os.path.join(save_path,'pred_on_t1')): os.mkdir(os.path.join(save_path,'pred_on_t1'))
        if not os.path.isdir(os.path.join(save_path,'pred_on_remapped')): os.mkdir(os.path.join(save_path,'pred_on_remapped'))
        if not os.path.isdir(os.path.join(save_path,'gt_on_t1')): os.mkdir(os.path.join(save_path,'gt_on_t1'))
        if not os.path.isdir(os.path.join(save_path,'flow')): os.mkdir(os.path.join(save_path,'flow'))
        if not os.path.isdir(os.path.join(save_path,'uncertainty')): os.mkdir(os.path.join(save_path,'uncertainty'))

        # temp viz start

        flow_gt1 = div_flow * flow_est_original[0].permute(1, 2, 0).detach().cpu().numpy()  # now shape is HxWx2
        flow_gt1 = cv2.resize(flow_vis.flow_to_color(flow_gt1), dsize=(640,480), interpolation=cv2.INTER_CUBIC)
        plt.imsave('{}/flow/epoch{}_batch{}.png'.format(save_path, epoch, batch), flow_gt1)
        flow_stds, std_map1 = calc_flow_std(flow_est_original)
        std_map1 = std_map1[0]
        std_map1 = std_map1.detach().cpu().clamp(min=0.0, max=5.0).numpy()
        std_map1 = cv2.resize(std_map1,dsize=(640,480),interpolation=cv2.INTER_NEAREST)
        plt.imsave('{}/uncertainty/epoch{}_batch{}.png'.format(save_path, epoch, batch),std_map1 )
        out_change_original1 = out_change_original[:,:,None].astype(np.bool8)
        out_change1 = overlay_result(out_change_original1,image_2.numpy(),color=None)
        out_change_remapped1 = overlay_result(out_change_original1,remapped_est, color=None)
        out_change1 = cv2.resize(out_change1,dsize=(640,480),interpolation=cv2.INTER_LINEAR)
        out_change_remapped1 = cv2.resize(out_change_remapped1,dsize=(640,480),interpolation=cv2.INTER_LINEAR)
        plt.imsave('{}/pred_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change1)
        plt.imsave('{}/pred_on_remapped/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change_remapped1)
        image_1_ = cv2.resize(image_1.numpy(),dsize=(640,480),interpolation=cv2.INTER_LINEAR)
        image_2_ = cv2.resize(image_2.numpy(),dsize=(640,480),interpolation=cv2.INTER_LINEAR)
        plt.imsave('{}/t0/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_1_)
        plt.imsave('{}/t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_2_)
        #temp viz end


        '''
        plt.imsave('{}/t0/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_1.numpy())
        plt.imsave('{}/t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),image_2.numpy())

        flow_gt = div_flow * flow_est_original[0].permute(1, 2, 0).detach().cpu().numpy()  # now shape is HxWx2
        plt.imsave('{}/flow/epoch{}_batch{}.png'.format(save_path, epoch, batch), flow_vis.flow_to_color(flow_gt))

        flow_stds, std_map = calc_flow_std(flow_est_original)
        std_map = std_map[0]
        plt.imsave('{}/uncertainty/epoch{}_batch{}.png'.format(save_path, epoch, batch), std_map.detach().cpu().clamp(min=0.0,max=5.0).numpy())


        if multi_class:
            out_change_color = seg_helper.classmap2colormap(torch.FloatTensor(out_change_original/50).cuda()).float().cpu().numpy()
            target_change_color = seg_helper.classmap2colormap(torch.FloatTensor(target_change_original/50).cuda()).float().cpu().numpy()
        else:
            out_change_color = None
            target_change_color = None
        out_change_original = out_change_original[:,:,None].astype(np.bool8)
        target_change_original = target_change_original[:,:,None].bool().numpy()
        out_change = overlay_result(out_change_original,image_2.numpy(),color=out_change_color)
        out_change_remapped = overlay_result(out_change_original,remapped_est, color=out_change_color)
        target_change_original = overlay_result(target_change_original,image_2.numpy(), color=target_change_color)
        plt.imsave('{}/pred_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change)
        plt.imsave('{}/pred_on_remapped/epoch{}_batch{}.png'.format(save_path, epoch, batch),out_change_remapped)
        plt.imsave('{}/gt_on_t1/epoch{}_batch{}.png'.format(save_path, epoch, batch),target_change_original)
        '''
        if return_img:
            return target_change_original.transpose(2,0,1)


    else:
        num_figs=8 if target_change_original is not None else 7
        fig, axis = plt.subplots(1, num_figs, figsize=(20, 10))
        axis[0].imshow(image_1.numpy())
        axis[0].set_title("Ref. Image")
        axis[0].axis('off')
        axis[1].imshow(image_2.numpy())
        axis[1].set_title("Query Image")
        axis[1].axis('off')

        # if apply_mask:
        #     mask = mask.detach()[0].cpu().numpy().astype(np.float32)
        # else:
        #     mask = np.ones((h_original, w_original))

        flow_stds, std_map = calc_flow_std(flow_est_original)
        flow_stds = flow_stds[0]
        std_map = std_map[0]

        flow_gt = div_flow * flow_est_original[0].permute(1, 2, 0).detach().cpu().numpy()  # now shape is HxWx2
        axis[2].imshow(flow_vis.flow_to_color(flow_gt))
        axis[2].set_title('Flow')
        axis[2].axis('off')

        std_map = axis[3].imshow(std_map.detach().cpu().clamp(min=0.0,max=5.0).numpy())
        # fig.colorbar(std_map)
        axis[3].set_title("Uncertainty (score={:.2f})".format(flow_stds))
        axis[3].axis('off')


        out_change_original_overlayed = overlay_result(out_change_original[:,:,None].astype(np.bool8),image_2.numpy(),color=None)
        axis[4].imshow(out_change_original_overlayed,vmax=255,interpolation='nearest')
        axis[4].set_title("Estim. on Query")
        axis[4].axis('off')
        remapped_est = overlay_result(out_change_original[:, :, None].astype(np.bool8),
                                                remapped_est, color=None)
        axis[5].imshow(remapped_est)
        axis[5].set_title("Estim. on Warped Ref.")
        axis[5].axis('off')
        
        axis[6].imshow(f_sim_map, cmap='jet')
        axis[6].set_title("Fused Feat. Sim.")
        axis[6].axis('off')

        if target_change_original is not None:
            target_change_original_overlayed = overlay_result(target_change_original[:, :, None].numpy().astype(np.bool8),
                                                    image_2.numpy(), color=None)
            axis[7].imshow(target_change_original_overlayed,vmax=255,interpolation='nearest')
            axis[7].set_title("GT on Query")
            axis[7].axis('off')

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
                loss_grid_weights=None,
                apply_mask=False,
                robust_l1_loss=False,
                sparse=False):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
        ssl: if True, change loss is replaced with SSL loss from the supervised loss
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
    running_cl_loss = 0
    running_feat_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        # pre-process the data
        source_image, target_image, source_image_256, target_image_256 = pre_process_data(
            mini_batch['source_image'],
            mini_batch['target_image'],
            device=device,
            norm=args.img_norm_type)
        source_change, target_change, source_change_256, target_change_256 = \
            pre_process_change(mini_batch['source_change'],
            mini_batch['target_change'],
            device=device)

        # when disable_flow =True, always warp with zero flow map
        disable_flow = mini_batch['disable_flow'][..., None, None].to(device)  # bs,1,1,1
        out_dict = net(source_image, target_image, source_image_256, target_image_256,disable_flow=disable_flow)
        out_flow_256, out_flow_orig = out_dict['flow']
        out_change_256, out_change_orig = out_dict['change']
        use_flow = mini_batch['use_flow'][...,None].to(device)
        # features for similarity loss, stop gradient to prevent detour flow decoder / half, quater, eight, sixteen
        feats_src, feats_tgt = out_dict['feature']
        feats_src, feats_tgt = [f.detach() for f in feats_src], [f.detach() for f in feats_tgt] 
        f12_src = [feats_src[0], feats_src[1], feats_src[2]]
        f12_tgt = [feats_tgt[0], feats_tgt[1], feats_tgt[2]]
        # At original resolutiont
        flow_gt_original = mini_batch['flow_map'].to(device)
        
        if flow_gt_original.shape[1] != 2:
            # shape is bxhxwx2
            flow_gt_original = flow_gt_original.permute(0,3,1,2)
        bs, _, h_original, w_original = flow_gt_original.shape
        weights_original = loss_grid_weights[-len(out_flow_orig):]

        # at 256x256 resolution, b, _, 256, 256
        if sparse:
            flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
        else:
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                        mode='bilinear', align_corners=False)
        flow_gt_256[:,0,:,:] *= 256.0/float(w_original)
        flow_gt_256[:,1,:,:] *= 256.0/float(h_original)
        bs, _, h_256, w_256 = flow_gt_256.shape
        weights_256 = loss_grid_weights[:len(out_flow_256)]

        # calculate the loss
        if args.usl:
            flow1, flow2, flow3, flow4 = out_flow_orig[-1], out_flow_orig[-2], out_flow_256[-1], out_flow_256[-2]
            cng1 = out_change_orig[-1]
            cng_prob1 = cng1.sigmoid()
        
            # # photometric loss, 0.01, 0.04, 0.19, 0.75 for 4-scale
            # loss_photo = ms_photo_loss(flows=[flow1], 
            #                            src_img= mini_batch['source_image'], tgt_img = mini_batch['target_image'], 
            #                            cng_masks=[cng_prob1],
            #                            weights=[1.], photometric='ssim', wavg=True)
            # loss_photo += ms_photo_loss(flows=[flow3, flow1], 
            #                            src_img= mini_batch['source_image'], tgt_img = mini_batch['target_image'], 
            #                            cng_masks=[cng_prob3, cng_prob1],
            #                            weights=[0.2, 0.8], photometric='census', wavg=True)
            
            loss_flow = multiscaleEPE(out_flow_orig, flow_gt_original, weights=weights_original, sparse=False,
                                    mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
            loss_flow += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                mean=True, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
            
            # flow smoothness regularization
            loss_flow_smooth = ms_smooth_reg(img=mini_batch['source_image'], outs=[flow1],
                                             edge_aware=True,
                                             weights=[1.])

            # fused feature similarity loss (shallow feat -> edge, many modes / deep feature -> abstract info., few modes)             
            ff_sim12, vmask = fused_feat_sim(f12_src, f12_tgt, flow1, h_original)
            # photo_diff_map, vmask_orig = calc_diff_map(flow1, mini_batch['source_image'].float().cuda(), mini_batch['target_image'].float().cuda(), photometric='robust')
            # loss_photo = mask_average(photo_diff_map*(1-use_flow.unsqueeze(1)), vmask_orig, wavg=True)
            # diff_map = F.interpolate(photo_diff_map, size=(ff_sim12.size()[-2], ff_sim12.size()[-1]), mode='bilinear')
            fused_diff = (1-ff_sim12.pow(1/3))
            loss_feat = mask_average(fused_diff*(1-use_flow.unsqueeze(1)), vmask*cng_prob1, wavg=True)
            # ff_sim12 as pseudo label, unsupervised focal loss for change mask
            # loss_change = -(spatial_centering(ff_sim12*(1-photo_diff_map), vmask, 0.3).detach()*cng_prob1).mean()*(1-use_flow.mean())
            # loss_change += -cng_prob1[~vmask].mean()
            loss_change = -0.125*((ff_sim12.pow(1/3)).detach()*(cng_prob1).log()*(1-use_flow.unsqueeze(1))).mean()
            # loss_change = 0.01*(1./torch.sin(cng_prob1.flatten(2).mean(-1, keepdim=True))*(1-use_flow)).mean()
 
            loss_total = loss_flow + loss_change + 0.05*loss_flow_smooth + 1.* loss_feat
        
        elif args.s_sl:
            pass
            
        else:
            if apply_mask:
                mask = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
                loss_flow = multiscaleEPE(out_flow_orig, flow_gt_original, weights=weights_original, sparse=sparse,
                                    mean=False, mask=mask, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
                if sparse:
                    mask_256 = sparse_max_pool(mask.unsqueeze(1).float(), (256, 256)).squeeze(1).byte() # bx256x256
                else:
                    mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                            align_corners=False).squeeze(1).byte() # bx256x256
                loss_flow += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256, sparse=sparse,
                                    mean=False, mask=mask_256, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
            elif args.except_occ:
                loss_flow = multiscaleEPE(mask*out_flow_orig, flow_gt_original, weights=weights_original, sparse=False,
                                    mean=False, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
                loss_flow += multiscaleEPE(mask*out_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                    mean=False, robust_L1_loss=robust_l1_loss,use_flow=use_flow)    
            else:
                loss_flow = multiscaleEPE(out_flow_orig, flow_gt_original, weights=weights_original, sparse=False,
                                    mean=False, robust_L1_loss=robust_l1_loss,use_flow=use_flow)
                loss_flow += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                    mean=False, robust_L1_loss=robust_l1_loss,use_flow=use_flow)

            loss_change = multiscaleCE(out_change_256,target_change_256,weights=weights_256)
            loss_change +=multiscaleCE(out_change_orig, target_change, weights=weights_original)     
            
            loss_total = loss_flow + loss_change   
        
        # contrastive loss
        if args.cl:
            cl = args.cl
            p1_list, p2_list, z1_list, z2_list = out_dict['pz'] # 1 indicies target, 2 indices source
            hws = [(int(h_original/4), int(w_original/4)),
                (int(h_original/8), int(w_original/8)),
                (32, 32),
                (16, 16)]
            flos = [F.interpolate(flow_gt_original, hws[i], mode='bilinear', align_corners=False) / h_original * hws[i][0]
                    for i in range(cl)] # scaled gt flow maps: /4, /8, 32, 16
            source_masks = [F.interpolate(source_change.float(), hws[i], mode='nearest') for i in range(cl)]
            target_masks = [F.interpolate(target_change.float(), hws[i], mode='nearest') for i in range(cl)]
            
            loss_cl = 0
            for p1, p2, z1, z2, flo, sm, tm in zip(p1_list, p2_list, z1_list, z2_list, flos, source_masks, target_masks):
                p1w = warp(p1, flo)
                z1w = warp(z1, flo)
                smaskw = warp(sm, flo)
                wmask = (p1w.norm(dim=1, keepdim=True) == 0).float()
                # occluded features are represented as 'True' in mask
                mask = (wmask + smaskw + tm).squeeze(dim=1).bool().detach() # B x H' x W'
        
                # calculate loss without occluded features
                loss_cl += -0.5 * ((F.cosine_similarity(p2, z1w, dim=1)
                                   + F.cosine_similarity(p1w, z2, dim=1))*use_flow)[~mask].mean()
        
            loss_total += loss_cl
                
        loss_total.backward()
        optimizer.step()

        running_total_loss += loss_total.item()
        running_flow_loss += loss_flow.item()
        running_change_loss += loss_change.item()
        running_cl_loss += loss_cl.item() if args.cl else 0.
        running_feat_loss += loss_feat.item() if args.usl else 0.

        writer.add_scalar('train_flow_per_iter', loss_flow.item(), n_iter)
        writer.add_scalar('train_change_per_iter', loss_change.item(), n_iter)
        writer.add_scalar('train_total_per_iter', loss_cl.item() if args.cl else 0., n_iter)
        writer.add_scalar('train_total_per_iter', loss_total.item(), n_iter)

        n_iter += 1
        msg = (f'loss_change:{running_change_loss/(i+1):.3f}/{loss_change.item():.3f} | '
               f'loss_flow:{running_flow_loss/(i+1):.3f}/{loss_flow.item():.3f} | '
               f'loss_feat:{running_feat_loss/(i+1):.3f}/{loss_feat.item():.3f} | '
               f'{cng_prob1.mean():.3f}/{cng_prob1.min():.3f} | '
               f'{ff_sim12[vmask].mean():.3f}')
        pbar.set_description(msg)
    running_total_loss /= len(train_loader)
    running_change_loss /= len(train_loader)
    running_flow_loss /= len(train_loader)
    running_cl_loss /= len(train_loader)


    return dict(total=running_total_loss, change=running_change_loss,
                cl=running_cl_loss, flow=running_flow_loss,
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

    running_total_loss = 0
    if not os.path.isdir(save_path): os.mkdir(save_path)

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)
        CE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)

        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                mini_batch['source_image'],
                mini_batch['target_image'],
                device=device,
                norm = args.img_norm_type)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(mini_batch['source_change'],
                                   mini_batch['target_change'],
                                   device=device)
            disable_flow = mini_batch['disable_flow'][..., None,None].to(device) # bs,1,1,1
            out_dict = net(source_image, target_image, source_image_256, target_image_256, disable_flow=disable_flow)
            out_flow_256, out_flow_orig = out_dict['flow']
            out_change_256, out_change_orig = out_dict['change']
            ''' Evaluate Flow '''
            # at original size
            flow_gt_original = mini_batch['flow_map'].to(device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape
            mask_gt = mini_batch['correspondence_mask'].to(device)
            weights_original = loss_grid_weights[-len(out_flow_orig):]

            # at 256x256 resolution, b, _, 256, 256
            if sparse:
                flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
            else:
                flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                            mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)
            bs, _, h_256, w_256 = flow_gt_256.shape
            weights_256 = loss_grid_weights[:len(out_flow_256)]

            if apply_mask:
                mask = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
                Loss = multiscaleEPE(out_flow_orig, flow_gt_original,
                                     weights=weights_original, sparse=sparse,
                                     mean=False, mask=mask, robust_L1_loss=robust_L1_loss)
                if sparse:
                    mask_256 = sparse_max_pool(mask.unsqueeze(1).float(), (256, 256)).squeeze(1).byte()  # bx256x256
                else:
                    mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                             align_corners=False).squeeze(1).byte()  # bx256x256
                Loss += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256,
                                      sparse=sparse,
                                      mean=False, mask=mask_256, robust_L1_loss=robust_L1_loss)
            else:
                Loss = multiscaleEPE(out_flow_orig, flow_gt_original,
                                     weights=weights_original, sparse=False,
                                     mean=False, robust_L1_loss=robust_L1_loss)
                Loss += multiscaleEPE(out_flow_256, flow_gt_256, weights=weights_256, sparse=False,
                                      mean=False, robust_L1_loss=robust_L1_loss)

            # calculating the validation EPE
            for index_reso_original in range(len(out_flow_orig)):
                EPE = div_flow * realEPE(out_flow_orig[-(index_reso_original+1)], flow_gt_original, mask_gt, sparse=sparse)
                EPE_array[index_reso_original, i] = EPE

            for index_reso_256 in range(len(out_flow_256)):
                EPE = div_flow * realEPE(out_flow_256[-(index_reso_256+1)], flow_gt_original, mask_gt,
                                        ratio_x=float(w_original) / float(256.0),
                                        ratio_y=float(h_original) / float(256.0),
                                        sparse=sparse)
                EPE_array[(len(out_flow_orig) + index_reso_256), i] = EPE
            # must be both in shape Bx2xHxW

            if i % 1000 == 0:
                vis_img = plot_during_training2(save_path, epoch, i, False,
                                               h_original, w_original, h_256, w_256,
                                               source_image, target_image, source_image_256, target_image_256, div_flow,
                                               flow_gt_original, flow_gt_256, output_net=out_flow_orig[-1],
                                               output_net_256=out_flow_256[-1],
                                               target_change_original=target_change,
                                               target_change_256=target_change_256,
                                               out_change_orig=out_change_orig,
                                               out_change_256=out_change_256,
                                               return_img=True)
                writer.add_image('val_warping_per_iter', vis_img, n_iter)

            # ''' Evaluate Change '''
            out_change_orig = torch.nn.functional.interpolate(out_change_orig[-1].detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            out_change_orig = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1)
            confmeter.add(out_change_orig, target_change.squeeze().long())

            running_total_loss += Loss.item()
            pbar.set_description(
                ' val total_loss: %.1f/%.1f' % (running_total_loss / (i + 1),
                                             Loss.item()))
        mean_epe = torch.mean(EPE_array, dim=1)

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU


    return dict(total=running_total_loss,mEPEs=mean_epe, accuracy = Acc,
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
    k = 2 if net.module.num_class == 1 else net.module.num_class
    confmeter = ConfusionMeter(k=k,normalized=False)

    net.eval()

    if not os.path.isdir(save_path): os.mkdir(save_path)
    print('Begin Testing {}'.format(save_path))
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                mini_batch['source_image'],
                mini_batch['target_image'],
                device=device,
                norm = args.img_norm_type)
            source_change, target_change, source_change_256, target_change_256 = \
                pre_process_change(mini_batch['source_change'],
                                   mini_batch['target_change'],
                                   device=device)
            disable_flow = mini_batch['disable_flow'][..., None,None].to(device) # bs,1,1,1
            out_dict = net(source_image, target_image, source_image_256, target_image_256, disable_flow=disable_flow)
            out_flow_256, out_flow_orig = out_dict['flow']

            out_change_256, out_change_orig = out_dict['change']

            bs, _, h_original, w_original = source_image.shape
            bs, _, h_256, w_256 = source_image_256.shape
            flow_gt_original = F.interpolate(out_flow_orig[-1], (h_original, w_original),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            flow_gt_256 = F.interpolate(out_flow_256[-1], (h_256, w_256),
                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
            
            feats_src, feats_tgt = out_dict['feature']
            feats_src, feats_tgt = [f.detach() for f in feats_src], [f.detach() for f in feats_tgt] 
            f12_src = [feats_src[0], feats_src[1], feats_src[2]]
            f12_tgt = [feats_tgt[0], feats_tgt[1], feats_tgt[2]]
            ff_sim1, vmask = fused_feat_sim(f12_src, f12_tgt, out_flow_orig[-1], h_original)
            # photo_diff_map, _ = calc_diff_map(out_flow_orig[-1], mini_batch['source_image'].float().cuda(), mini_batch['target_image'].float().cuda(), photometric='robust')
            # photo_diff_map = F.interpolate(photo_diff_map, size=(ff_sim1.size()[-2], ff_sim1.size()[-1]), mode='bilinear')

            if i % plot_interval == 0:
                plot_during_training2(save_path, epoch, i, False,
                                        h_original, w_original, h_256, w_256,
                                        source_image, target_image, source_image_256, target_image_256, div_flow,
                                        flow_gt_original, flow_gt_256, output_net=out_flow_orig[-1],
                                        output_net_256=out_flow_256[-1],
                                        target_change_original=target_change,
                                        target_change_256=target_change_256,
                                        out_change_orig=out_change_orig,
                                        out_change_256=out_change_256,
                                        f_sim_map = (1-ff_sim1.pow(1/3))*vmask, vmask=vmask,
                                        return_img=False,
                                        multi_class=net.module.num_class)
                
            out_change_orig = torch.nn.functional.interpolate(out_change_orig[-1].detach(),
                                                              size=(h_original, w_original), mode='bilinear')
            vmask = torch.nn.functional.interpolate(vmask.float(), size=(h_original, w_original), mode='nearest')
            
            if net.module.num_class == 1:
                out_change = (torch.sigmoid(out_change_orig)).le(0.5).long() * vmask
                out_change = out_change.permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
            else:
                out_change = out_change_orig.permute(0, 2, 3, 1).reshape(-1, out_change_orig.shape[1])
            
            target_change = target_change.long() * vmask
            target_change = target_change.detach().permute(0, 2, 3, 1).reshape(-1, 1).squeeze()
            try:
                confmeter.add(out_change, target_change)
            except:
                print(out_change_orig.shape)
                print(target_change.shape)
                print(net.module.num_class)
                print(out_change_orig.max())

    conf = torch.FloatTensor(confmeter.value())
    Acc = 100*(conf.diag().sum() / conf.sum()).item()
    recall = conf[1,1]/(conf[1,0]+conf[1,1])
    precision =conf[1,1]/(conf[0,1]+conf[1,1])
    f1 = 100*2*recall*precision/(recall+precision)
    IoUs, mIoU = IoU(conf)
    IoUs, mIoU = 100 * IoUs, 100 * mIoU

    return dict(accuracy = Acc, IoUs=IoUs, mIoU = mIoU, f1=f1)
