import torch
#import the network base class in torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################

'''
* @name: weights_init_normal
* @description: initialize the parameters of the given neural network normally
* @param m: the input neural network
''' 
def weights_init_normal(m):
    #get the name the network
    classname = m.__class__.__name__
    # print(classname)

    #see which type of network
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

'''
* @name: weights_init_xavier
* @description: initialize the parameters of the given neural network by xavier method
* @param m: the input neural network
''' 
def weights_init_xavier(m):
    #get the name the network
    classname = m.__class__.__name__
    # print(classname)
    #see which type of network
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

'''
* @name: weights_init_kaiming
* @description: initialize the parameters of the given neural network by kaiming method
* @param m: the input neural network
''' 
def weights_init_kaiming(m):
    #get the name the network
    classname = m.__class__.__name__
    # print(classname)

    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

'''
* @name: weights_init_kaiming
* @description: initialize the parameters of the given neural network by kaiming method
* @param m: the input neural network
''' 
def weights_init_orthogonal(m):
    #get the name the network
    classname = m.__class__.__name__
    print(classname)

    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

'''
* @name: init_weights
* @description: initialize the parameters of the given neural network
* @param net: the input neural network
* @init_type: select the initialize type from (normal, xavier, kaiming, orthogonal)
''' 
def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    #call corresponding function based on the init_type
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

'''
* @name: get_norm_layer
* @description: create the norm_layer
* @param norm_type: select the norm_layer type from ('instance' and 'batch')
* @return norm_layer: the norm layer
''' 
def get_norm_layer(norm_type='instance'):
    #call the corresponding function based on the norm_type
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    #return norm_layer
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

'''
* @name: define_G
* @description: define the Generative neural network based on configuration
* @param input_nc: channel number of input image
* @param output_nc: channel number of output image
* @param ngf: number of generator filters in first convolutional layer
* @param which_model_netG: select model for Generator
* @param norm: set instance normalization or batch normalization
* @param use_dropout: set dropout for the generator
* @param init_type: set the initialization method(normal, xavier, kaiming orthogonal)
* @return netG: the generative neural network
''' 
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal'):
    netG = None
    #get the norm_layer
    norm_layer = get_norm_layer(norm_type=norm)
    #create generator based on selected model type
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    #initialize the parameters of the generators
    init_weights(netG, init_type=init_type)
    #return the neural network of generator
    return netG

'''
* @name: define_D
* @description: define the Discriminative neural network based on configuration
* @param input_nc: channel number of input image
* @param ndf: number of discriminator filters in the first convolutional layer
* @param which_model_netD: select model for Discriminator
* @param n_layers_D: when the model of discriminator is n_layers, set the layer numbers
* @param norm: set instance normalization or batch normalization
* @param use_sigmoid: whether to use a sigmoid in discriminator
* @param init_type: set the initialization method(normal, xavier, kaiming orthogonal)
* @return netD: the discriminator neural network
''' 
def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal'):
    netD = None
    #get the norm_layer
    norm_layer = get_norm_layer(norm_type=norm)
    #create generator based on selected model type
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'global':
        netD = GlobalDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'global_np':
        netD = GlobalNPDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    #initialize the parameters of the discriminator
    init_weights(netD, init_type=init_type)
    #return the neural network of the discriminator
    return netD

'''
* @name: print_network
* @description: print the parameters of the input network
* @param net: the input network
'''   
def print_network(net):
    #initialize the number of the parameters
    num_params = 0
    #loop all parameters to add up the number
    for param in net.parameters():
        num_params += param.numel()
    #print the network
    print(net)
    #print the total number of the network
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

#the GANLoss class is derived from torch.nn.Module
class GANLoss(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of the class
    * @param use_lsgan: the flag the use which loss function
    * @param target_real_label: the label for a good image
    * @param target_fake_label: the label for a bad image
    * @param device: the device to do the calculation
    '''   
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 device='cpu'):
        #...
        super(GANLoss, self).__init__()
        #save the device
        self.device = device
        #save the label for a good image
        self.real_label = target_real_label
        #save the label for a good image
        self.fake_label = target_fake_label
        #save the real_label_var and fake_label_var as None by default
        self.real_label_var = None
        self.fake_label_var = None
        #decide the loss the function
        if use_lsgan:
            #if use_lsgan, then use MSELoss function
            self.loss = nn.MSELoss().to(device)
        else:
            #if not use_lsgan, then use BCELoss function
            self.loss = nn.BCELoss().to(device)
    '''
    * @name: get_target_tensor
    * @description: the constructor of the class
    * @param input: the input image ???
    * @param target_is_real: if the target is a real image
    * @return: return the label, which is a tensor
    ''' 
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        #if tht target is real
        if target_is_real:
            #the flag of whether to create a label
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            #if need to create a new label
            if create_label:
                #call torch.full to create a label for real images
                self.real_label_var = torch.full(input.size(), self.real_label, requires_grad=False, device=self.device)
            #save the label value
            target_tensor = self.real_label_var
        else:
            #if the target is fake
            #the flag of whether to create a label
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            #if need to create a new label
            if create_label:
                #call torch.full to create a label for fake images
                self.fake_label_var = torch.full(input.size(), self.fake_label, requires_grad=False, device=self.device)
            target_tensor = self.fake_label_var
        #return the label, which is a tensor
        return target_tensor
    '''
    * @name: __call__
    * @description: ...
    * @param input: the input image ???
    * @param target_is_real: if the target is a real image
    * @return: return the label, which is a tensor
    ''' 
    def __call__(self, input, target_is_real):
        #get the target tensor(or the label of the input image)
        target_tensor = self.get_target_tensor(input, target_is_real)
        #calculate the value of loss function
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

#the ResnetGenerator class is derived from torch.nn.Module
class ResnetGenerator(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of the class
    * @param input_nc: channel number of input image
    * @param output_nc: channel number of output image
    * @param ngf: number of generator filters in first convolutional layer
    * @param norm_layer: the norm_layer
    * @param use_dropout: set dropout for the generator
    * @param n_blocks: set the number of blocks
    * @param padding_type: select the padding type from 
    '''   
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        #get number of channels of input image
        self.input_nc = input_nc
        #set the number of channels of output layer
        self.output_nc = output_nc
        #set the number of filters in the fisrt convolutional layer
        self.ngf = ngf

        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #create the model, consist with a ReflectionPad2d, a Conv2d, a norm_layer and a ReLU
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        #set the number of downsampling
        n_downsampling = 2
        #add Conv2d layer for downsampling to the network
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        #???
        mult = 2**n_downsampling
        #add a ResnetBlock
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        #add ConvTranspose2d layer 
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        #add ReflectionPad2d layer 
        model += [nn.ReflectionPad2d(3)]
        #add Conv2d layer 
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #add a Tanh layer
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of the ResnetBlock
    * @param dim:???
    * @param padding_type: select the padding type from
    * @param norm_layer: the norm_layer
    * @param use_dropout: set dropout for the generator
    * @param use_bias: set whether to use bias
    '''       
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        #build the block
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    '''
    * @name: build_conv_block
    * @description: build ResnetBlock
    * @param dim:???
    * @param padding_type: select the padding type from
    * @param norm_layer: the norm_layer
    * @param use_dropout: set dropout for the generator
    * @param use_bias: set whether to use bias
    '''    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []

        p = 0
        #add a layer based on the padding type
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        #add a Conv2d layer, a norm_layer and a ReLU layer
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        #add a dropout layer
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0)]

        p = 0
        #add another padding padding layer
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        #add a Conv2d layer and a norm_layer
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        #build the block
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of the UnetGenerator
    * @param input_nc: channel number of input image
    * @param output_nc: channel number of output image
    * @param num_downs: ???
    * @param ngf: number of generator filters in first convolutional layer
    * @param norm_layer: the norm_layer
    * @param use_dropout: set dropout for the generator
    '''   
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of the UnetSkipConnectionBlock
    * @param outer_nc: ???
    * @param inner_nc: ???
    * @param input_nc: ???
    * @param submodule: ???
    * @param outermost: ???
    * @param innermost: ???
    * @param norm_layer: the norm_layer
    * @param use_dropout: set dropout for the generator
    '''   
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        #set if it`s outermost
        self.outermost = outermost
        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        
        #get a Conv2d layer    
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        #get a LeakyReLU layer
        downrelu = nn.LeakyReLU(0.2, True)
        #get a norm_layer
        downnorm = norm_layer(inner_nc)
        #get a ReLU layer
        uprelu = nn.ReLU(True)
        #get a norm layer
        upnorm = norm_layer(outer_nc)

        #if is outermost
        if outermost:
            #get a ConvTranspose2d layer
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            #get the pre-defined Conv2d layer
            down = [downconv]
            #get the pre-defined ReLU, ConvTranspose2d, Tanh layer
            up = [uprelu, upconv, nn.Tanh()]
            #combine up and down together and the submodule if any
            model = down + [submodule] + up
        #if is the innermost
        elif innermost:
            #get a ConvTranspose2d layer
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            #the down contains a ReLU and a Conv2d layer             
            down = [downrelu, downconv]
            #the up contains pre-defined ReLU, ConvTranspose2d and norm_layer
            up = [uprelu, upconv, upnorm]
            #combine down and up
            model = down + up
        else:
            #get a ConvTranspose2d layer
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            #the down contains a ReLU , a Conv2d and a norm_layer                            
            down = [downrelu, downconv, downnorm]
            #the up contains pre-defined ReLU, ConvTranspose2d and norm_layer
            up = [uprelu, upconv, upnorm]
            #add drop-out layer or not
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of NLayerDiscriminator
    * @param input_nc: channel number of input image
    * @param ndf: number of discriminator filters in first convolutional layer
    * @param n_layers: ??? number of hidden layers
    * @param norm_layer: set instance normalization or batch normalization
    * @param use_sigmoid: whether to use sigmoid
    ''' 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = 4
        padw = 1

        #add a Conv2d layer and LeakyReLU layer
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # 128

        nf_mult = 1
        nf_mult_prev = 1
        #add several combinations of a Conv2d, a norm_layer and a LeakyReLU layer
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # 64
        # 32

        #add a Conv2d, a norm_layer and a LeakyReLU layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 31
        #add a Conv2d layer
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # 30

        #add a sigmoid or not based on input
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GlobalDiscriminator(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of GlobalDiscriminator
    * @param input_nc: channel number of input image
    * @param ndf: number of discriminator filters in first convolutional layer
    * @param n_layers: ??? number of hidden layers
    * @param norm_layer: set instance normalization or batch normalization
    * @param use_sigmoid: whether to use sigmoid
    ''' 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(GlobalDiscriminator, self).__init__()
        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = 4
        padw = 1
        #add a Conv2d layer and LeakyReLU layer
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # 128
        nf_mult = 1
        nf_mult_prev = 1
        #add several combinations of a Conv2d, a norm_layer and a LeakyReLU layer
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # 64
        # 32

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        #add a Conv2d, a norm_layer and a LeakyReLU layer
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 16
        #add a Conv2d
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=0)]
        #add a Conv2d
        sequence += [nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0)]
        #add a sigmoid or not based on input
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GlobalNPDiscriminator(nn.Module):
    # no padding
    '''
    * @name: __init__
    * @description: the constructor of GlobalNPDiscriminator
    * @param input_nc: channel number of input image
    * @param ndf: number of discriminator filters in first convolutional layer
    * @param n_layers: ??? number of hidden layers
    * @param norm_layer: set instance normalization or batch normalization
    * @param use_sigmoid: whether to use sigmoid
    ''' 
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(GlobalNPDiscriminator, self).__init__()
        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 256
        kw = [8, 3, 4]
        padw = 0
        #add a Conv2d layer and LeakyReLU layer
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw[0], stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        # 125
        nf_mult = 1
        nf_mult_prev = 1
        #add several combinations of a Conv2d, a norm_layer and a LeakyReLU layer
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw[n], stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # 62
        # 30

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        #add a Conv2d, a norm_layer and a LeakyReLU layer
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # 14
        #add a Conv2d
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=2, padding=0)]
        # 6
        #add a Conv2d
        sequence += [nn.Conv2d(1, 1, kernel_size=6, stride=1, padding=0, bias=use_bias)]
        # 1
        #add a sigmoid or not based on input
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PixelDiscriminator(nn.Module):
    '''
    * @name: __init__
    * @description: the constructor of PixelDiscriminator
    * @param input_nc: channel number of input image
    * @param ndf: number of discriminator filters in first convolutional layer
    * @param n_layers: ??? number of hidden layers
    * @param norm_layer: set instance normalization or batch normalization
    * @param use_sigmoid: whether to use sigmoid
    ''' 
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        #set use_bias flag based on the type of the norm_layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #add a Conv2d, a LeakyReLU, a Conv2d, norm_layer, a leakyReLU and a Conv2d
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        #add a sigmoid or not based on input
        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

