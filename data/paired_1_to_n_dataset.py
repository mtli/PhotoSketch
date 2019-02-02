import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from data.rotate_and_crop import rotate_and_crop

class Paired1ToNDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.if_aug = opt.phase == 'train'
        self.img_dir = os.path.join(self.root, 'image')
        self.skt_dir = os.path.join(self.root, opt.render_dir, opt.aug_folder)
        list_file = os.path.join(self.root, 'list', opt.phase + '.txt')
        with open(list_file) as f:
            content = f.readlines()
        self.list = sorted([x.strip() for x in content])

    def __getitem__(self, index):
        N = self.opt.nGT
        filename = self.list[index]

        if_crop_or_rotate = self.if_aug and ((self.opt.crop or self.opt.rotate) and random.random() < 0.8)
        # even with augmentation on, 20% chance does nothing
        if_flip = self.if_aug and ((not self.opt.no_flip) and random.random() < 0.5)

        pathA = os.path.join(self.img_dir, filename + '.jpg')
        img = Image.open(pathA)
        fine_size = self.opt.fineSize

        if if_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.if_aug and self.opt.color_jitter:
            jitter_amount = 0.02
            img = transforms.ColorJitter(jitter_amount, jitter_amount, jitter_amount, jitter_amount)(img)
        if if_crop_or_rotate:
            load_size = self.opt.loadSize
            img = img.resize((load_size, load_size), Image.BICUBIC)
            if self.opt.rotate:
                rot_deg = 5*random.randint(-3, 3)
                img = rotate_and_crop(img, rot_deg, True)
            if self.opt.crop:
                img = transforms.ToTensor()(img)
                w_offset = random.randint(0, max(0, load_size - fine_size - 1))
                h_offset = random.randint(0, max(0, load_size - fine_size - 1))
                img = img[:, h_offset:h_offset + fine_size,
                          w_offset:w_offset + fine_size]
            else:
                img = img.resize((fine_size, fine_size), Image.BICUBIC)
                img = transforms.ToTensor()(img)
        else:
            img = img.resize((fine_size, fine_size), Image.BICUBIC)
            img = transforms.ToTensor()(img)

        if self.opt.inverse_gamma:
            linear_mask = (img <= 0.04045).float()
            exponential_mask = (img > 0.04045).float()
            img = (img / 12.92 * linear_mask) + (((img + 0.055) / 1.055) ** 2.4) * exponential_mask

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        B = N*[None]
        for i in range(N):
            pathB = os.path.join(self.skt_dir, '%s_%02d.png' % (filename, i+1))
            img = Image.open(pathB)
            if if_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if if_crop_or_rotate:
                load_size = self.opt.loadSize
                img = img.resize((load_size, load_size), Image.BICUBIC)
                if self.opt.rotate:
                    img = rotate_and_crop(img, rot_deg, True)
                if self.opt.crop:
                    img = transforms.ToTensor()(img)
                    img = img[:, h_offset:h_offset + fine_size,
                              w_offset:w_offset + fine_size]
                else:
                    img.resize((fine_size, fine_size), Image.BICUBIC)
                    img = transforms.ToTensor()(img)
            else:
                img = img.resize((fine_size, fine_size), Image.BICUBIC)
                img = transforms.ToTensor()(img)

            B[i] = transforms.Normalize((0.5,), (0.5,))(img)

        B = torch.cat(B, 0)

        return {'A': A, 'B': B,
                'A_paths': pathA, 'B_paths': pathB}

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'Paired1ToNDataset'

