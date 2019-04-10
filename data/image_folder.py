###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

import platform


#get the valid extenstion name
if platform.system() == 'Windows':
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
    ]
else:
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]

'''
* @name: is_image_file
* @description: check if a input file is an image
* @param filename: the name of the file
* @return: whether the file is an image
'''
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

'''
* @name: make_dataset
* @description: save all images in the given directory in a set
* @param dir: the path of the directory
* @return: the set which saves the paths of all images
'''
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

'''
* @name: default_loader
* @description: convert the image to RGB format
* @param path: the path of the given image
* @return: the image in RGB format
'''
def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    '''
    * @name: __init__
    * @description: initialize the class
    * @param root: the path of the directory
    * @param transform: the method of the transform of the image, default to be None
    * @param return_paths: the flag to return the path instead of the image, default to be False
    * @param loader: the format of the image, default to be RGB
    '''
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
    '''
    * @name: __getitem__
    * @description: get the image from the image set based on the given index
    * @param index: the index of the image
    * @return: return the transformed image
    '''
    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img
    '''
    * @name: __len__
    * @description: get the length of the image set
    * @return: the length of the image set
    '''
    def __len__(self):
        return len(self.imgs)
