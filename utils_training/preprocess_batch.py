import numpy as np
import torch

def post_process_single_img_data(source_img, target_img, norm='z_score', color_order ='rgb'):
    # resolution original
    if norm=='z_score':
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=source_img.dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=source_img.dtype).view(3, 1, 1)
        image_1 = (source_img.detach().cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)*255
        image_2 = (target_img.detach().cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)*255
        return image_1, image_2

    elif norm == 'min_max':
        image_1 = (source_img.detach().cpu().permute(1,2,0)+1) * 128
        image_2 = (target_img.detach().cpu().permute(1,2,0)+1) * 128
        return image_1, image_2



def pre_process_data(source_img, target_img, device, norm='z_score', rgb_order = 'rgb'):
    '''
    Pre-processes source and target images before passing it to the network
    :param source_img: Torch tensor Bx3xHxW
    :param target_img: Torch tensor Bx3xHxW
    :param device: cpu or gpu
    :return:
    source_img_copy: Torch tensor Bx3xHxW, source image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    target_img_copy: Torch tensor Bx3xHxW, target image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    source_img_256: Torch tensor Bx3x256x256, source image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    target_img_256: Torch tensor Bx3x256x256, target image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    '''
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape
    if rgb_order == 'bgr':
        source_img = source_img[:,[2,1,0]]
        target_img = target_img[:,[2,1,0]]
    if norm == 'z_score':
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])

        # original resolution
        source_img_copy = source_img.float().to(device).div(255.0)
        target_img_copy = target_img.float().to(device).div(255.0)
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

        # resolution 256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()

        source_img_256 = source_img_256.float().div(255.0)
        target_img_256 = target_img_256.float().div(255.0)
        source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
    elif norm == 'min_max':
        # original resolution
        source_img_copy = source_img.float().to(device).div(128.0) - 1.0
        target_img_copy = target_img.float().to(device).div(128.0) - 1.0
        # resolution 256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()
        source_img_256 = source_img_256.float().div(128.0) - 1.0
        target_img_256 = target_img_256.float().div(128.0) - 1.0

    else:
        raise KeyError

    return source_img_copy, target_img_copy, source_img_256, target_img_256

def pre_process_change(source_mask, target_mask, device):
    '''
    Pre-processes source and target images before passing it to the network
    :param source_mask: Torch tensor BxHxW
    :param target_mask: Torch tensor BxHxW
    :param device: cpu or gpu
    :return:
    source_img_copy: Torch tensor Bx1xHxW, source image
    target_img_copy: Torch tensor Bx1xHxW, target image
    source_img_256: Torch tensor Bx1x256x256, source image rescaled to 256x256
    target_img_256: Torch tensor Bx1x256x256, target image rescaled to 256x256
    '''
    # img has shape bxhxw
    b, h_scale, w_scale = target_mask.shape
    # original resolution
    source_img_copy = source_mask.long().to(device)[:,None,...]
    target_img_copy = target_mask.long().to(device)[:,None,...]

    # resolution 256x256
    source_img_256 = torch.nn.functional.interpolate(input=source_mask.float().to(device)[:,None,...],
                                                     size=(256, 256),
                                                     mode='nearest')
    target_img_256 = torch.nn.functional.interpolate(input=target_mask.float().to(device)[:,None,...],
                                                     size=(256, 256),
                                                     mode='nearest')

    source_img_256 = source_img_256.long()
    target_img_256 = target_img_256.long()

    return source_img_copy, target_img_copy, source_img_256, target_img_256