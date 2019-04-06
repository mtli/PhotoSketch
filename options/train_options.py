from .base_options import BaseOptions

#Inherited from base_option class
class TrainOptions(BaseOptions):
    def initialize(self):
        #initialize the base class data members
        BaseOptions.initialize(self)
        #set the frequency of showing training result on screen
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        #the numbers of image displayed in a row
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        #set the frequency of saving training results to html
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        #set the frequency of showing training results on console
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        #set the frequency of saving the latest results
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        #set the frequency of saving checkpoints at the end of epochs
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        #set the whether to continue training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        #set the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        #set the type of program(test, train, val)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        #set which epoch to load? set to latest to use latest cached model
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #set number of iterations at starting learing rate
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        #set number of iterations to linearly decay learning rate to zero
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        #set momentum term of adam
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        #set initial learning rate for adam
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        #set whether not to use least square GAN, if false, use vanilla GAN
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        #set weight for cycle loss (A -> B -> A)
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #set weight for cycle loss (B -> A -> B)
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        #set weight for stroke generation
        self.parser.add_argument('--lambda_stroke', type=float, default=100, help='weight for stroke generation')
        #set weight for GAN in loss function
        self.parser.add_argument('--lambda_G', type=float, default=1, help='weight for GAN')
        #set the size of image buffer that stores previously generated images
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        #set flag that do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        #set learning rate policy: lambda|step|plateau
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        #multiply the learning by a gamma every lr_decay_iters iterations
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        #set identity mapping loss factor
        self.parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        #set isTrain by true
        self.isTrain = True
