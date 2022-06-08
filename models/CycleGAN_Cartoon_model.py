import torch
import torch.nn as nn
import torch.nn.functional as nf
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import junyanz_networks as networks
from models.pretrainnet import VGGPreTrained
from typing import List, Tuple
from losses.GreyscaleLoss import GreyscaleLoss
from losses.EdgeLoss import EdgeLoss
import numpy as np

class GuidedFilter(nn.Module):
  def box_filter(self, x: torch.Tensor, r):
    ch = x.shape[1]
    k = 2 * r + 1
    weight = 1 / ((k)**2)  # 1/9
    # [c,1,3,3] * 1/9
    box_kernel = torch.ones((ch, 1, k, k), dtype=torch.float32, device=x.device).fill_(weight)
    # same padding
    return nf.conv2d(x, box_kernel, padding=r, groups=ch)

  def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
    b, c, h, w = x.shape
    device = x.device
    # 全1的图像进行滤波的结果
    N = self.box_filter(torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

    mean_x = self.box_filter(x, r) / N
    mean_y = self.box_filter(y, r) / N
    cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
    var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = self.box_filter(A, r) / N
    mean_b = self.box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output


class ColorShift(nn.Module):
  def __init__(self, mode='uniform'):
    super().__init__()
    self.dist: Distribution = None
    self.mode = mode

  def setup(self, device: torch.device):
    # NOTE 原论文输入的bgr图像，此处需要改为rgb
    if self.mode == 'normal':
      self.dist = torch.distributions.Normal(
          torch.tensor((0.299, 0.587, 0.114), device=device),
          torch.tensor((0.1, 0.1, 0.1), device=device))
    elif self.mode == 'uniform':
      self.dist = torch.distributions.Uniform(
          torch.tensor((0.199, 0.487, 0.014), device=device),
          torch.tensor((0.399, 0.687, 0.214), device=device))

  #Allow taking mutiple images batches as input
  #So we can do: gray_fake, gray_cartoon = ColorShift(output, input_cartoon)
  def forward(self, *image_batches: torch.Tensor) -> Tuple[torch.Tensor]:
    # Sample the random color shift coefficients
    weights = self.dist.sample()

    # images * self.weights[None, :, None, None] => Apply weights to r,g,b channels of each images
    # torch.sum(, dim=1) => Sum along the channels so (B, 3, H, W) become (B, H, W)
    # .unsqueeze(1) => add back the channel so (B, H, W) become (B, 1, H, W)
    # .repeat(1, 3, 1, 1) => (B, 1, H, W) become (B, 3, H, W) again
    return ((((torch.sum(images * weights[None, :, None, None], dim= 1)) / weights.sum()).unsqueeze(1)).repeat(1, 3, 1, 1) for images in image_batches)

def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

class junyanzCycleGANCartoonModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'tv_A', 'tv_B', 'feat_A2B', 'feat_B2A', 'D_B_gray', 'D_B_blur', 'gray_A', 'gray_B', 'edge_A', 'edge_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.is_train and self.opt['lambda_identity'] > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.is_train:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_B_gray', 'D_B_blur']
            self.network_names = ['netG_A', 'netG_B', 'netD_A', 'netD_B', 'netD_B_gray', 'netD_B_blur']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            self.network_names = ['netG_A', 'netG_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt['input_nc'], opt['output_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])
        self.netG_B = networks.define_G(opt['output_nc'], opt['input_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])

        if self.is_train:  # define discriminators
            self.vgg_pretrained = VGGPreTrained().to(self.device)
            self.vgg_pretrained.setup(self.device)
            self.guided_filter = GuidedFilter()
            self.color_shift = ColorShift()
            self.color_shift.setup(self.device)
            self.netD_A = networks.define_D(opt['output_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])
            self.netD_B = networks.define_D(opt['input_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])
            self.netD_B_gray = networks.define_D(opt['input_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])
            self.netD_B_blur = networks.define_D(opt['input_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])

        if self.is_train:
            if opt['lambda_identity'] > 0.0:  # only works when input and output images have the same number of channels
                assert(opt['input_nc'] == opt['output_nc'])
            self.fake_A_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt['gan_mode']).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.tv_loss = total_variation_loss
            self.feat_loss = torch.nn.L1Loss()
            self.greyscale_loss = GreyscaleLoss()
            self.edge_loss = EdgeLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt['lr'], betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_B_gray.parameters(), self.netD_B_blur.parameters()), lr=opt['lr'], betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt['direction'] == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_B_gray(self):
        """Calculate GAN loss for discriminator D_B_gray"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        gray_fake_B, gray_real_B = self.color_shift(fake_B, self.real_B)
        self.loss_D_B_gray = self.backward_D_basic(self.netD_B_gray, gray_real_B, gray_fake_B)

    def backward_D_B_blur(self):
        """Calculate GAN loss for discriminator D_B_blur"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        blur_fake_B, blur_real_B = self.guided_filter(fake_B, fake_B, r=5), self.guided_filter(self.real_B, self.real_B, r=5)
        self.loss_D_B_blur = self.backward_D_basic(self.netD_B_blur, blur_real_B, blur_fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt['lambda_identity']
        lambda_A = self.opt['lambda_A']
        lambda_B = self.opt['lambda_B']
        lambda_tv = self.opt['lambda_tv']
        lambda_feat = self.opt['lambda_feat']
        lambda_gray = self.opt['lambda_gray']
        lambda_edge = self.opt['lambda_edge']
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # GAN loss D_B_gray(G_B(B))
        gray_fake_B, = self.color_shift(self.fake_B)
        self.loss_G_B_gray = self.criterionGAN(self.netD_B_gray(gray_fake_B), True)
        # GAN loss D_B_gray(G_B(B))
        blur_fake_B = self.guided_filter(self.fake_B, self.fake_B, r=5)
        self.loss_G_B_blur = self.criterionGAN(self.netD_B_blur(blur_fake_B), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Total variation loss
        self.loss_tv_A = self.tv_loss(self.fake_A) * lambda_tv
        self.loss_tv_B = self.tv_loss(self.fake_B) * lambda_tv
        # Feature loss
        b, c, h, w = self.real_A.shape
        vgg_fake_B = self.vgg_pretrained(self.fake_B)
        vgg_real_A = self.vgg_pretrained(self.real_A)
        self.loss_feat_A2B = (self.feat_loss(vgg_real_A, vgg_fake_B)/(b*c*h*w)) * lambda_feat
        vgg_fake_A = self.vgg_pretrained(self.fake_A)
        vgg_real_B = self.vgg_pretrained(self.real_B)
        self.loss_feat_B2A = (self.feat_loss(vgg_real_B, vgg_fake_A)/(b*c*h*w)) * lambda_feat
        # Grayscale loss
        self.loss_gray_A = self.greyscale_loss(self.rec_A, self.real_A) * lambda_gray
        self.loss_gray_B = self.greyscale_loss(self.rec_B, self.real_B) * lambda_gray
        # Edge loss
        self.loss_edge_A = self.edge_loss(self.rec_A, self.real_A) * lambda_edge
        self.loss_edge_B = self.edge_loss(self.rec_B, self.real_B) * lambda_edge
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_tv_A + self.loss_tv_B + self.loss_feat_A2B + self.loss_feat_B2A + self.loss_gray_A + self.loss_edge_A + self.loss_gray_B + self.loss_edge_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_B_gray, self.netD_B_blur], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_B_gray, self.netD_B_blur], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_B_gray()
        self.backward_D_B_blur()
        self.optimizer_D.step()  # update D_A and D_B's weights
