import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):  

        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--logs', action='store_true', help='logs loss during traiing or not')
        self.parser.add_argument('--savelog_dir', type=str, default='./Logs',help='save log directory')
        self.parser.add_argument('--MyTest', type=str, required=True,default='NA',help='Here is MyTest [ Normal | Diff | Spec | Rough | ALL_4D | ALL_1D | ALL_1D_Render | ALL_5D_Render | NA || L1]')
        self.parser.add_argument('--mode', type=str, default='Syn',help='Syn or Real mode')

        # input settings       
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--rand_light', type=float, default=0.5, help='random light range for augment input images')                
        self.parser.add_argument('--augment_input', action='store_true', help='data augmentation')
        self.parser.add_argument('--dataroot', type=str, default='F:/LoganZhou/Research/dataset/DeepMaterialsData/ImageData/SynTrainData', help='path of loading traing data or testing data') 
        self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--LowCam', action='store_true', help='postiion of camera: not Low Camera: [0,0,2.14]; Low Camera: [0,0,1]')        

        # generator
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output feature maps (diff, normal, rough) channels')
        self.parser.add_argument('--input_nc_D', type=int, default=3, help='# of input image channels of Discriminator')
        self.parser.add_argument('--rough_nc', type=int, default=1, help='# of output roughness channel')
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG: [ global | newarch | VA_Net]')

        self.initialized = True

    def parse(self, seed,save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('seed %s' % str(seed))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)

        if save:# and not self.opt.continue_train:
            if self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt_resume.txt')
            else:
                file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('seed %s\n' % str(seed))
                opt_file.write('-------------- End ----------------\n')

        # if save and self.opt.load_pretrain:
        #     file_name = os.path.join(expr_dir, 'opt.txt')
        #     with open(file_name, 'wt') as opt_file:
        #         opt_file.write('------------ Options -------------\n')
        #         for k, v in sorted(args.items()):
        #             opt_file.write('%s: %s\n' % (str(k), str(v)))
        #         opt_file.write('seed %s\n' % str(seed))
        #         opt_file.write('-------------- End ----------------\n')

        return self.opt
