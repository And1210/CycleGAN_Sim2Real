import torchvision
import torch
import torch.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
import pytorch_lightning as pl
from typing import Sequence, List, Tuple, Union

def normalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return (im - mean) / std

def denormalize(im: Union[np.ndarray, torch.Tensor], mean=0.5, std=0.5):
  return im * std + mean

class PretrainNet(pl.LightningModule):
  def train(self, mode: bool):
    return super().train(False)

  def state_dict(self, destination, prefix, keep_vars):
    destination = OrderedDict()
    destination._metadata = OrderedDict()
    return destination

  def setup(self, device: torch.device):
    self.freeze()

class VGGPreTrained(PretrainNet):
  def __init__(self, output_index: int = 26):
    """pytorch vgg pretrained net

    Args:
        output_index (int, optional): output layers index. Defaults to 26.
        NOTE the finally output layer name is `output_index-1`
        ```
          (0): Conv2d (1): ReLU
          (2): Conv2d (3): ReLU
          (4): MaxPool2d
          (5): Conv2d (6): ReLU
          (7): Conv2d (8): ReLU
          (9): MaxPool2d
          (10): Conv2d (11): ReLU
          (12): Conv2d (13): ReLU
          (14): Conv2d (15): ReLU
          (16): Conv2d (17): ReLU
          (18): MaxPool2d
          (19): Conv2d (20): ReLU
          (21): Conv2d (22): ReLU
          (23): Conv2d (24): ReLU
          (25): Conv2d (26): ReLU
          (27): MaxPool2d
          (28): Conv2d (29): ReLU
          (30): Conv2d (31): ReLU
          (32): Conv2d (33): ReLU
          (34): Conv2d (35): ReLU
          (36): MaxPool2d
        ```
    """
    super().__init__()
    vgg = torchvision.models.vgg19(pretrained=True)
    self.features = vgg.features
    self.output_index = output_index
    del vgg

  def _process(self, x):
    # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
    return self.vgg_normalize(denormalize(x))

  def setup(self, device: torch.device):
    mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    self.vgg_normalize = lambda x: normalize(x, mean, std)
    self.freeze()

  def _forward_impl(self, x):
    x = self._process(x)
    # See note [TorchScript super()]
    # NOTE get output with out relu activation
    x = self.features[:self.output_index](x)
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)
    return x

  def forward(self, x):
    return self._forward_impl(x)
