import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt

class BinnedRGBLoss(nn.Module):
    def __init__(self, bin_factor, device):
        super().__init__()

        self.bin_factor = bin_factor
        self.mse_loss = nn.MSELoss()
        self.PI = np.pi

        self.device = device

    def bin_img(self, img):
        output = torch.zeros(img.shape).to(self.device)
        binned = torch.mul(img, self.bin_factor)
        for i in range(self.bin_factor):
            output = torch.add(output, torch.special.expit(torch.mul(torch.sub(binned, i), 50)))
        output = torch.div(output, self.bin_factor)
        return output

    # def bin_img(self, img):
    #     binned = torch.mul(img, self.bin_factor)
    #     # binned = torch.floor(binned)
    #     binned = torch.sub(binned, torch.div(torch.sin(torch.mul(binned, 2*self.PI)), 2*self.PI))
    #     binned = torch.div(binned, self.bin_factor)
    #     return binned

    def __call__(self, output, target):
        out_bin = self.bin_img(output)
        trg_bin = self.bin_img(target)

        # out_img = output[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_img = target[0].permute(1, 2, 0).cpu().detach().numpy()
        # out_binned = out_bin[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_binned = trg_bin[0].permute(1, 2, 0).cpu().detach().numpy()
        #
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(out_img)
        # axs[0, 1].imshow(trg_img)
        # axs[1, 0].imshow(out_binned)
        # axs[1, 1].imshow(trg_binned)
        # plt.show()
        # # plt.waitforbuttonpress()
        # plt.close()

        loss = self.mse_loss(out_bin, trg_bin)
        return loss
