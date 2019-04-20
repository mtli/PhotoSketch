from .base_options import BaseOptions

#Inherited from base_option class
class TestOptions(BaseOptions):
    def initialize(self):
        #initialize the base class data members
        BaseOptions.initialize(self)

        #set the number of test examples
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        #set the directory to save the results
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        #set the aspect ratio of result image
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        #set the type of program(test, train, val)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #set which epoch to load
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #set the number of test images
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        #set the file name
        self.parser.add_argument('--file_name', type=str, default='')
        #set the suffix
        self.parser.add_argument('--suffix', type=str, default='')
        #set isTrain by false
        self.isTrain = False
