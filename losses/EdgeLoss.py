import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from kornia.filters import sobel
import matplotlib.pyplot as plt

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse_loss = nn.MSELoss()

    def gradient_img(self, img):
        return sobel(img, normalized=False)

    def old_gradient_img(self, img):
        img = torch.mean(input=img, dim=1, keepdim=True)
        x = img

        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # conv1 = F.conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        weight1 = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0)).cuda()
        # conv1 = conv1.cuda()
        # G_x = conv1(Variable(x)).data
        G_x = F.conv2d(x, weight1, padding=1)

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # conv2 = F.conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        weight2 = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0)).cuda()
        # conv2 = conv2.cuda()
        # G_y = conv2(Variable(x)).data
        G_y = F.conv2d(x, weight2, padding=1)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G

    def __call__(self, output, target):
        out_grad = self.gradient_img(output)
        trg_grad = self.gradient_img(target)

        # out_img = output[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_img = target[0].permute(1, 2, 0).cpu().detach().numpy()
        # out_edge = out_grad[0].permute(1, 2, 0).cpu().detach().numpy()
        # trg_edge = trg_grad[0].permute(1, 2, 0).cpu().detach().numpy()
        #
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(out_img)
        # axs[0, 1].imshow(trg_img)
        # axs[1, 0].imshow(out_edge)
        # axs[1, 1].imshow(trg_edge)
        # plt.show()
        # # plt.waitforbuttonpress()
        # plt.close()

        loss = self.mse_loss(out_grad, trg_grad)
        return loss
