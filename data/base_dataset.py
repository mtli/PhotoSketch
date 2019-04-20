#An abstract class representing a Dataset,All subclasses should override __len__, that provides the size of the dataset, and __getitem__, supporting integer indexing in range from 0 to len(self) exclusive
import torch.utils.data as data
#The Image module provides a class with the same name which is used to represent a PIL image. The module also provides a number of factory functions, including functions to load images from files, and to create new images.
from PIL import Image
#
import torchvision.transforms as transforms

#base class of data set
class BaseDataset(data.Dataset):
    def __init__(self):

        #initialize the base class
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

'''
* @name: get_transform
* @description: the function to scale or crop the input function
* @param opt: the parameter set
* @return: the composed transform method of an image
'''
def get_transform(opt):
    #the list the save every single transform on input images
    transform_list = []
    #if the scaling and cropping of images method is set by resize_and_crop
    if opt.resize_or_crop == 'resize_and_crop':
        #get the set image size
        osize = [opt.loadSize, opt.loadSize]
        #save resized result
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        #save cropped randomly result
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    #if crop only
    elif opt.resize_or_crop == 'crop':
        #crop randomly
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    #if scale only
    elif opt.resize_or_crop == 'scale_width':
        #scale the image
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    #if scale with crop
    elif opt.resize_or_crop == 'scale_width_and_crop':
        #save scaled result
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        #save cropped randomly result
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    #see if need to add flipped images
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())


    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    #Composes several transforms together, and return the new transform
    return transforms.Compose(transform_list)
'''
* @name: __scale_width
* @description: scale the input image based on the width
* @param img: the input image
* @param target_width: the target width after scaling
* @return: return the image after scaling
'''
def __scale_width(img, target_width):
    #get the original size of image
    ow, oh = img.size
    if (ow == target_width):
        return img
    #calculate the new width and height
    w = target_width
    h = int(target_width * oh / ow)
    #scale the image
    return img.resize((w, h), Image.BICUBIC)
