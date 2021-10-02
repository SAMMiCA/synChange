import torch
import torch.nn.functional as F

SMOOTH = 1e-6


# Expect outputs and labels to have same shape (ie: torch.Size([batch:1, 224, 224])), and type long
def iou_segmentation(outputs: torch.Tensor, labels: torch.Tensor):
    # Will be zero if Truth=0 or Prediction=0
    intersection = (outputs & labels).float().sum((1, 2))
    # Will be zzero if both are 0
    union = (outputs | labels).float().sum((1, 2))

    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

def EPE(input_flow, target_flow, sparse=False, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/batch_size


def L1_loss(input_flow, target_flow):
    L1 = torch.abs(input_flow-target_flow)
    L1 = torch.sum(L1, 1)
    return L1


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=[0.1,1.0,1.0,1.0,1.0], size_average=True,device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device=device
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha]).to(device)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha).to(device)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def L1_charbonnier_loss(input_flow, target_flow, sparse=False, mean=True, sum=False):

    batch_size = input_flow.size(0)
    epsilon = 0.01
    alpha = 0.4
    L1 = L1_loss(input_flow, target_flow)
    norm = torch.pow(L1 + epsilon, alpha)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        norm = norm[~mask]
    if mean:
        return norm.mean()
    elif sum:
        return norm.sum()
    else:
        return norm.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, robust_L1_loss=False, mask=None, weights=None,
                  sparse=False, mean=False):
    '''
    here the ground truth flow is given at the higest resolution and it is just interpolated
    at the different sized (without rescaling it)
    :param network_output:
    :param target_flow:
    :param weights:
    :param sparse:
    :return:
    '''

    def one_scale(output, target, sparse, robust_L1_loss=False, mask=None, mean=False):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))

            if mask is not None:
                mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            target_scaled = F.interpolate(target, (h, w), mode='bilinear', align_corners=False)

            if mask is not None:
                # mask can be byte or float or uint8 or int
                # resize first in float, and then convert to byte/int to remove the borders
                # which are values between 0 and 1
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).byte()
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

        if robust_L1_loss:
            if mask is not None:
                return L1_charbonnier_loss(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
            else:
                return L1_charbonnier_loss(output, target_scaled, sparse, mean=mean, sum=False)
        else:
            if mask is not None:
                return EPE(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
            else:
                return EPE(output, target_scaled, sparse, mean=mean, sum=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, target_flow, sparse, robust_L1_loss=robust_L1_loss, mask=mask, mean=mean)
    return loss


def realEPE(output, target, mask_gt, ratio_x=None, ratio_y=None, sparse=False, mean=True, sum=False):
    '''
    in this case real EPE, the network output is upsampled to the size of
    the target (without scaling) because it was trained without the scaling, it should be equal to target flow
    mask_gt can be uint8 tensor or byte or int
    :param output:
    :param target: flow in range [0, w-1]
    :param sparse:
    :return:
    '''
    # mask_gt in shape bxhxw, can be torch.byte or torch.uint8 or torch.int
    b, _, h, w = target.size()
    if ratio_x is not None and ratio_y is not None:
        upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
        upsampled_output[:,0,:,:] *= ratio_x
        upsampled_output[:,1,:,:] *= ratio_y
    else:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    # output interpolated to original size (supposed to be in the right range then)

    flow_target_x = target.permute(0, 2, 3, 1)[:, :, :, 0]
    flow_target_y = target.permute(0, 2, 3, 1)[:, :, :, 1]
    flow_est_x = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 0]  # BxH_xW_
    flow_est_y = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 1]

    flow_target = \
        torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                   flow_target_y[mask_gt].unsqueeze(1)), dim=1)
    flow_est = \
        torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                   flow_est_y[mask_gt].unsqueeze(1)), dim=1)
    return EPE(flow_est, flow_target, sparse, mean=mean, sum=sum)


def multiscaleCE(network_output, target_change, mask=None, weights=None, criterion = FocalLoss()):
    '''
    here the ground truth flow is given at the higest resolution and it is just interpolated
    at the different sized (without rescaling it)
    :param network_output:
    :param target_flow:
    :param weights:
    :param sparse:
    :return:
    '''

    def one_scale(output, target, mask=None):
        b, _, h, w = output.size()

        target_scaled = F.interpolate(target.float(), (h, w), mode='nearest')

        if mask is not None:
            # mask can be byte or float or uint8 or int
            # resize first in float, and then convert to byte/int to remove the borders
            # which are values between 0 and 1
            mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='nearest', align_corners=False).byte()
            mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

        if mask is not None:
            return criterion(output * mask.float(), (target_scaled * mask.float()).long())
        else:
            return criterion(output, target_scaled.long())

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, target_change,mask=mask)
    return loss
