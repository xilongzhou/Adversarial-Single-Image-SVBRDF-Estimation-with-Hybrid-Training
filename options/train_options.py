from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--test_dataroot', type=str, default='F:/LoganZhou/Research/dataset/DeepMaterialsData/ImageData/SynTestData3') 

        # for output and display
        self.parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')        

        # for training setttings
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for adam')
        self.parser.add_argument('--lambda_sr_gan', type=float, default=1.0, help='weight for L1 loss of spec feature map')                
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--lambda_l1', type=float, default=10.0, help='weight for l1 loss')                
        self.parser.add_argument('--lambda_reall1', type=float, default=10.0, help='weight for l1 loss')                
        self.parser.add_argument('--lambda_render', type=float, default=5.0, help='weight for render discriminator loss')                
        self.parser.add_argument('--lambda_real', type=float, default=1.0, help='weight for real training')  
        self.parser.add_argument('--no_l1_loss', action='store_true', help='if specified, do *not* use VGG l1 loss')        
        self.parser.add_argument('--no_reall1_loss', action='store_true', help='if specified, do *not* use l1 loss for real image')        
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')

        # for real images train
        self.parser.add_argument('--In_scale', action='store_true', help='invariant scale or not for real images training')                
        self.parser.add_argument('--real_train', action='store_true', help='real image training or not') 
        self.parser.add_argument('--gan_realA', action='store_true', help='realA for GAN or not') 
        self.parser.add_argument('--L1_realA', action='store_true', help='realA for L1 or not') 
        self.parser.add_argument('--vg_realA', action='store_true', help='realA for vgg or not') 
        self.parser.add_argument('--no_real_vgg_loss', action='store_true', help='use for vgg or not for real training') 
        self.parser.add_argument('--real_dataroot', type=str, default='F:/LoganZhou/Research/dataset/DeepMaterialsData/ImageData/RealTrainData2_Linear',help='real images training dataset') 
        self.parser.add_argument('--real_batchSize', type=int, default=1, help='input batch size of real images')
        self.parser.add_argument('--loadSize_real', type=int, default=256, help='scale real images to 256 (default)')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization for discriminators')        
        self.parser.add_argument('--use_dropout_D', action='store_true', help='use dropout for the Discriminator')

        self.isTrain = True
