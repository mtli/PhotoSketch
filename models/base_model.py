import os
import torch

'''
* @name: BaseModel
* @description: the base class of model used in this paper
'''  
class BaseModel():
    '''
    * @name: name
    * @description: return the name of this model
    * @return: the name of this model
    '''   
    def name(self):
        return 'BaseModel'
    '''
    * @name: initialize
    * @description: initialize the model with the parameter set
    * @param opt: the configured parameter set
    '''   
    def initialize(self, opt):
        #save the parameter set in self.opt
        self.opt = opt
        #save the flag of the type of the program(train or test) in self.isTrain
        self.isTrain = opt.isTrain
        #save the calculation device, default to be "CPU"
        self.device = torch.device("cuda" if opt.use_cuda else "cpu")
        #save the directory to save the results
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    '''
    * @name: set_input
    * @description: the virtual function to save the input
    * @param input: the input of the model
    '''  
    def set_input(self, input):
        self.input = input
    '''
    * @name: forward
    * @description: the virtual function to ...
    '''  
    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass
    '''
    * @name: get_image_paths
    * @description: the virtual function to get the path storing the images
    '''  
    def get_image_paths(self):
        pass
    '''
    * @name: optimize_parameters
    * @description: the virtual function to ...
    '''  
    def optimize_parameters(self):
        pass
    '''
    * @name: get_current_visuals
    * @description: the virtual function to ...
    '''  
    def get_current_visuals(self):
        return self.input
    '''
    * @name: get_current_errors
    * @description: the virtual function to ...
    '''  
    def get_current_errors(self):
        return {}
    '''
    * @name: get_current_errors
    * @description: the virtual function to ...
    '''  
    def save(self, label):
        pass
    '''
    * @name: save_network
    * @description: the virtual function to save the network
    '''  
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network = network.to(self.device)
    '''
    * @name: load_network
    * @description: the virtual function to load the network
    '''  
    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if self.opt.pretrain_path:
            save_path = os.path.join(self.opt.pretrain_path, save_filename)
        else:
            save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
    '''
    * @name: update_learning_rate
    * @description: update the learning rate, called once in every epoch
    '''  
    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
