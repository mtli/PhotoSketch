import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from data.rotate_and_crop import rotate_and_crop

#the class inheriated from BaseDataSet for 1 to n case
class Paired1ToNDataset(BaseDataset):
    '''
    * @name: initialize
    * @description: initialize the class based on input parameter set
    * @param opt: the parameter set
    '''
    def initialize(self, opt):
        #set the opt by input parameter set
        self.opt = opt
        #set the directory of input data
        self.root = opt.dataroot
        #set the flag for whether to use data augumentation based on if the progamming is running under train mode
        self.if_aug = opt.phase == 'train'
        #set the dir for images of the training data
        self.img_dir = os.path.join(self.root, 'image')
        #set the dir for corresponding sketches of the training 
        self.skt_dir = os.path.join(self.root, opt.render_dir, opt.aug_folder)
        #set the path of list_file
        list_file = os.path.join(self.root, 'list', opt.phase + '.txt')
        #read in the info in the list_file
        with open(list_file) as f:
            content = f.readlines()
        #set the list
        self.list = sorted([x.strip() for x in content])
    '''
    * @name: __getitem__
    * @description: get the data of selected index
    * @param index: the index of data
    * @return: the information of the selected data
    '''   
    def __getitem__(self, index):
        #set the number of sketches corresonding to an image
        N = self.opt.nGT
        #get the file name based on input index
        filename = self.list[index]
        #the flag to see whether to crop of rotate for data augmentation
        # even with augmentation on, 20% chance does nothing
        if_crop_or_rotate = self.if_aug and ((self.opt.crop or self.opt.rotate) and random.random() < 0.8)
        
        #the flag to see whether to flip for data augmentation
        # even with augmentation on, 50% chance does nothing
        if_flip = self.if_aug and ((not self.opt.no_flip) and random.random() < 0.5)

        #get the path of jpg image
        pathA = os.path.join(self.img_dir, filename + '.jpg')
        #read in the image
        img = Image.open(pathA)
        #get the fine size
        fine_size = self.opt.fineSize

        #if need flip, then flip from left to right
        if if_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #if need color jitter
        if self.if_aug and self.opt.color_jitter:
            jitter_amount = 0.02
            img = transforms.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(img)
        #if need crop or rotate
        if if_crop_or_rotate:
            #resize the original image
            load_size = self.opt.loadSize
            img = img.resize((load_size, load_size), Image.BICUBIC)

            #if rotate
            if self.opt.rotate:
                #generate the rotation degree randomly
                rot_deg = 5*random.randint(-3, 3)
                #roate the image
                img = rotate_and_crop(img, rot_deg, True)
            #if crop
            if self.opt.crop:
                #turn image to tensor
                img = transforms.ToTensor()(img)
                #get width and height offset randomly
                w_offset = random.randint(0, max(0, load_size - fine_size - 1))
                h_offset = random.randint(0, max(0, load_size - fine_size - 1))
                #crop the image
                img = img[:, h_offset:h_offset + fine_size,
                          w_offset:w_offset + fine_size]
            else:
                img = img.resize((fine_size, fine_size), Image.BICUBIC)
                img = transforms.ToTensor()(img)
        else:
            #just turn the img to matrix? or tensor
            img = img.resize((fine_size, fine_size), Image.BICUBIC)
            img = transforms.ToTensor()(img)
        
        #if inverse_gamma???
        if self.opt.inverse_gamma:
            linear_mask = (img <= 0.04045).float()
            exponential_mask = (img > 0.04045).float()
            img = (img / 12.92 * linear_mask) + (((img + 0.055) / 1.055) ** 2.4) * exponential_mask

        #normalize the image
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        #get corresponding list of sketches for the image
        B = N*[None]
        #loop all sketches
        for i in range(N):
            #get path
            pathB = os.path.join(self.skt_dir, '%s_%02d.png' % (filename, i+1))
            #read in sketch in png
            img = Image.open(pathB)
            #if need flip, then flip from left to right
            if if_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #if need crop or rotate    
            if if_crop_or_rotate:
                #resize the original image
                load_size = self.opt.loadSize
                img = img.resize((load_size, load_size), Image.BICUBIC)
                #if rotate
                if self.opt.rotate:
                    #rotate sketch with same angle
                    img = rotate_and_crop(img, rot_deg, True)
                #if rotate
                if self.opt.crop:
                    #crop with same offset
                    img = transforms.ToTensor()(img)
                    img = img[:, h_offset:h_offset + fine_size,
                              w_offset:w_offset + fine_size]
                else:
                    #just turn the img to matrix? or tensor
                    img.resize((fine_size, fine_size), Image.BICUBIC)
                    img = transforms.ToTensor()(img)
            else:
                #just turn the img to matrix? or tensor
                img = img.resize((fine_size, fine_size), Image.BICUBIC)
                img = transforms.ToTensor()(img)
            #normalize the sketch
            B[i] = transforms.Normalize((0.5,), (0.5,))(img)
        #save B
        B = torch.cat(B, 0)

        return {'A': A, 'B': B,
                'A_paths': pathA, 'B_paths': pathB}
    '''
    * @name: __len__
    * @description: get the length of the data set
    * @return: length of the data set
    '''
    def __len__(self):
        return len(self.list)
    '''
    * @name: name
    * @description: return the name of the object
    * @return: return the name
    '''
    def name(self):
        return 'Paired1ToNDataset'

