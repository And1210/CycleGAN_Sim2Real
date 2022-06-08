import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt

class GreyscaleLoss(nn.Module):
    def __init__(self, radius=2, method="var"):
        super().__init__()

        self.mse_loss = nn.MSELoss()

    def greyscale(self, img):
        return torch.mean(img, dim=1)

    def __call__(self, output, target):
        out_grey = self.greyscale(output)
        trg_grey = self.greyscale(target)

        # out_img = output[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_img = target[0].permute(1, 2, 0).cpu().detach().numpy()
        # out_edge = out_lbp[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_edge = trg_lbp[0].permute(1, 2, 0).cpu().detach().numpy()

        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(out_img)
        # axs[0, 1].imshow(trg_img)
        # axs[1, 0].imshow(out_edge)
        # axs[1, 1].imshow(trg_edge)
        # plt.show()
        # # plt.waitforbuttonpress()
        # plt.close()

        loss = self.mse_loss(out_grey, trg_grey)
        return loss
