import os.path
from glob import glob
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, IMG_EXTENSIONS
from PIL import Image


class TestDirDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.in_dir = opt.dataroot
        if opt.file_name:
            self.list = [os.path.join(self.in_dir, opt.file_name)]
        else:
            self.list = []
            for ext in IMG_EXTENSIONS:
                self.list.extend(glob(os.path.join(self.in_dir, '*' + ext)))

    def __getitem__(self, index):
        file_path = self.list[index]
        img = Image.open(file_path)
        w, h = img.size
        fine_size = self.opt.fineSize
        img = img.resize((fine_size, fine_size), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        B = A.clone()[0]

        return {'A': A, 'B': B, 'A_paths': file_path, 'B_paths': file_path, 'w': w, 'h': h}

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'TestDirDataset'

