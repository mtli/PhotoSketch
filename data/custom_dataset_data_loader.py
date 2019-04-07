import torch.utils.data
from data.base_data_loader import BaseDataLoader

'''
* @name: CreateDataset
* @description: scale the input image based on the width
* @param opt: the parameter set
* @return: return the data set
'''
def CreateDataset(opt):
    dataset = None
    #if the mode is "1_to_n"
    if opt.dataset_mode == '1_to_n':
        from data.paired_1_to_n_dataset import Paired1ToNDataset
        #call class Paired1ToNDataset
        dataset = Paired1ToNDataset()
    #if the mode is "test_dir"
    elif opt.dataset_mode == 'test_dir':
        from data.test_dir_dataset import TestDirDataset
        #call class TestDirDataset
        dataset = TestDirDataset()
    #error msg
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    #initialize the dataset, basically process data augumentation
    dataset.initialize(opt)

    return dataset

#the class inheriated from BaseDataLoader
class CustomDatasetDataLoader(BaseDataLoader):
    
    '''
    * @name: name
    * @description: return the name of the object
    * @return: return the name
    '''
    def name(self):
        return 'CustomDatasetDataLoader'
    '''
    * @name: initialize
    * @description: initialize the BaseDataLoader
    * @param opt: the parameter set
    '''
    def initialize(self, opt):
        #initialize the base class
        BaseDataLoader.initialize(self, opt)
        #set the dataset
        self.dataset = CreateDataset(opt)
        #set the dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
    '''
    * @name: initialize
    * @description: initialize the BaseDataLoader
    * @param opt: the parameter set
    '''
    def load_data(self):
        return self

    '''
    * @name: __len__
    * @description: get the length of the data set
    * @return: length of the data set
    '''
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    '''
    * @name: __iter__
    * @description: the iterator of the data set
    * @return: the iterator of the data set
    '''
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
