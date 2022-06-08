import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

class TextureLoss(nn.Module):
    def __init__(self, radius=2, method="var"):
        super().__init__()

        self.radius = radius
        self.n_points = 8*radius
        self.method = method

        self.mse_loss = nn.MSELoss()

    def lbp_transform(self, imgs):
        out = torch.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            for j in range(imgs.shape[1]):
                new_imgs = imgs[i,j,:,:].cpu().detach().numpy()
                new_imgs = local_binary_pattern(new_imgs, self.n_points, self.radius, self.method)
                new_imgs = np.nan_to_num(new_imgs)
                new_imgs = torch.from_numpy(new_imgs).float().to(0)
                out[i,j,:,:] = new_imgs

        return out

    def __call__(self, output, target):
        out_lbp = self.lbp_transform(output)
        trg_lbp = self.lbp_transform(target)

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

        loss = self.mse_loss(out_lbp, trg_lbp)
        return loss
