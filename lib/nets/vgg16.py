# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

class MELM_vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = 512
    self._fc7_channels = 4096

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    # Remove fc8
    self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.vgg.features[layer].parameters(): p.requires_grad = False

    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

    # ------- parallel Gpu-----------
    self._layers['head'] = torch.nn.DataParallel(self._layers['head'])

  def _image_to_headtest(self):
    net_conv = self._layers['head'](self._image)
    return net_conv

  def _image_to_head(self):
    if self.training:
      net_conv = []
      for i in range(2):
        net_conv.append(self._layers['head'](self._image[i]))
    else:
      net_conv = self._layers['head'](self._image)
    return net_conv

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)
    self._predictions['fc7'] = fc7

    return fc7

  def load_pretrained_cnn(self, state_dict):
    self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})
    #self.vgg.classifier = torch.nn.DataParallel(self.vgg.classifier)


