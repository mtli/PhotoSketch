#argparse is a python library for command-line interfaces
import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        #the path of training data
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        #batch size
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        #scale input images to this size
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        #crop image to this size
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        #channel number of input image
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        #channel number of output image
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        #number of generator filters in first convolutional layer
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        #number of discriminator filters in the first convolutional layer
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        #select model for Discriminator
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        #select model for Generator
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        #when the model of discriminator is n_layers, set the layer numbers
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        #disable the CUDA training
        self.parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training (please use CUDA_VISIBLE_DEVICES to select GPU)')
        #set the name of the experiment, if decides where to store samples and models
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        #set the data_set mode(unaligned, aligned, and single)
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        #set the model(cycle_gan, pix2pix, test)
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        #set the direction
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        #set the number of threads for loading data
        self.parser.add_argument('--nThreads', default=6, type=int, help='# threads for loading data')
        #set the directory to save models
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        #set instance normalization or batch normalization
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        #set the order (serail or random) to input images 
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        #set the display window size
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        #set the window id of web display
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        #set the server of the web display
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        #set the port  of the web display
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        #set dropout for the generator
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        #set the maximum number of samples allowed from the dataset
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        #downscal method(scaling or cropping)
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        #set whether use flip to do data augmentation
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        #set the initialization method(normal, xavier, kaiming orthogonal)
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        #set the directory to store sketch rendered images
        self.parser.add_argument('--render_dir', type=str, default='sketch-rendered')
        self.parser.add_argument('--aug_folder', type=str, default='width-5')
        self.parser.add_argument('--stroke_dir', type=str, default='')

        # set whether to crop the input image
        self.parser.add_argument('--crop', action='store_true')
        # set whether to rotate the input image
        self.parser.add_argument('--rotate', action='store_true')
        # set whether to color jitter the input image
        self.parser.add_argument('--color_jitter', action='store_true')
        self.parser.add_argument('--stroke_no_couple', action='store_true', help='')
        self.parser.add_argument('--pretrain_path', type=str, default='')

        # set the number of sketches corresonding to an image
        self.parser.add_argument('--nGT', type=int, default=5)
        self.parser.add_argument('--rot_int_max', type=int, default=3)
        # set the color jitter amount if needed
        self.parser.add_argument('--jitter_amount', type=float, default=0.02)
        self.parser.add_argument('--inverse_gamma', action='store_true')
        self.parser.add_argument('--img_mean', type=float, nargs='+')
        self.parser.add_argument('--img_std', type=float, nargs='+')
        # set the path of the list_file
        self.parser.add_argument('--lst_file', type=str)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        
        #save the parameters in opt
        self.opt = self.parser.parse_args()
        #set the isTrain flag in opt
        self.opt.isTrain = self.isTrain   # train or test
        #set the use_cuda flag in opt based on os and configuration
        self.opt.use_cuda = not self.opt.no_cuda and torch.cuda.is_available()

        #cast opt to a dictionary
        args = vars(self.opt)

        #print all parameters
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk

        #get the dir to save the model by connecting checkpoints_dir and name
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        #save the parameters in opt.txt file
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
