import torch
import torch.nn as nn
from models.DR_TANet.util import upsample
from models.DR_TANet.TANet_element import *

class TANet(nn.Module):

    def __init__(self, encoder_arch='resnet18', local_kernel_size=1, stride=1, padding=0, groups=4, drtam=True, refinement=True, num_class=2):
        super(TANet, self).__init__()

        self.encoder1, channels = get_encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=True)
        self.attention_module = get_attentionmodule(local_kernel_size, stride, padding, groups, drtam, refinement, channels)
        self.decoder = get_decoder(channels=channels)
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.num_class = num_class

    def forward(self, im_target, im_source, im_target_256, im_source_256,disable_flow=None):
        img_t1 = im_target
        img_t0 = im_source
        # img_t0,img_t1 = torch.split(img,3,1)
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)
        features = features_t0 + features_t1
        features_map = self.attention_module(features)
        pred_ = self.decoder(features_map)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.bn(pred_)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.relu(pred_)
        pred = self.classifier(pred_)

        return {
            'change': pred
        }

