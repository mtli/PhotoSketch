import os.path
from glob import glob
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, IMG_EXTENSIONS
from PIL import Image

#the class inheriated from BaseDataSet for test case
class TestDirDataset(BaseDataset):
    '''
    * @name: initialize
    * @description: initialize the class based on input parameter set
    * @param opt: the parameter set
    '''
    def initialize(self, opt):
        #set the opt by input parameter set
        self.opt = opt
        #set the directory of input data
        self.in_dir = opt.dataroot
        #get all images in that directory
        if opt.file_name:
            self.list = [os.path.join(self.in_dir, opt.file_name)]
        else:
            self.list = []
            for ext in IMG_EXTENSIONS:
                self.list.extend(glob(os.path.join(self.in_dir, '*' + ext)))
    '''
    * @name: __getitem__
    * @description: get the data of selected index
    * @param index: the index of data
    * @return: the information of the selected data
    '''   
    def __getitem__(self, index):
        #get the path of the image
        file_path = self.list[index]
        #read in the image
        img = Image.open(file_path)
        #get the size of the image
        w, h = img.size
        #get the predefined size of image
        fine_size = self.opt.fineSize
        #resize the image
        img = img.resize((fine_size, fine_size), Image.BICUBIC)
        #tranform the image based on predefined tranforms commands
        img = transforms.ToTensor()(img)
        #normalize the image
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        #B...
        B = A.clone()[0]

        return {'A': A, 'B': B, 'A_paths': file_path, 'B_paths': file_path, 'w': w, 'h': h}
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
        return 'TestDirDataset'

