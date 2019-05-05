import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from PIL import Image


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        self.netG = self.netG.to(self.device)
        if self.isTrain:
            self.netD = self.netD.to(self.device)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, device=self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if 'w' in input:
            self.input_w = input['w']
        if 'h' in input:
            self.input_h = input['h']

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self.netG(self.real_A)
        self.real_B = self.input_B

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).detach())
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        n = self.real_B.shape[1]
        loss_D_real_set = torch.empty(n, device=self.device)
        for i in range(n):
            sel_B = self.real_B[:, i, :, :].unsqueeze(1)
            real_AB = torch.cat((self.real_A, sel_B), 1)
            pred_real = self.netD(real_AB)
            loss_D_real_set[i] = self.criterionGAN(pred_real, True)
        self.loss_D_real = torch.mean(loss_D_real_set)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_G

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        # Second, G(A) = B
        n = self.real_B.shape[1]
        fake_B_expand = self.fake_B.expand(-1, n, -1, -1)
        L1 = torch.abs(fake_B_expand - self.real_B)
        L1 = L1.view(-1, n, self.real_B.shape[2]*self.real_B.shape[3])
        L1 = torch.mean(L1, 2)
        min_L1, min_idx = torch.min(L1, 1)
        self.loss_G_L1 = torch.mean(min_L1) * self.opt.lambda_A
        self.min_idx = min_idx

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach())
        fake_B = util.tensor2im(self.fake_B.detach())
        if self.isTrain:
            sel_B = self.real_B[:, self.min_idx[0], :, :]
        else:
            sel_B = self.real_B[:, 0, :, :]
        real_B = util.tensor2im(sel_B.unsqueeze(1).detach())
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label)
        self.save_network(self.netD, 'D', label)

    def write_image(self, out_dir):
        image_numpy = self.fake_B.detach()[0][0].cpu().float().numpy()
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        image_pil = image_pil.resize((self.input_w[0], self.input_h[0]), Image.BICUBIC)
        name, _ = os.path.splitext(os.path.basename(self.image_paths[0]))
        out_path = os.path.join(out_dir, name + self.opt.suffix + '.png')
        image_pil.save(out_path)
       