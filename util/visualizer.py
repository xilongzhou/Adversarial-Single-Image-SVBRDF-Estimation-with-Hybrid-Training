import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        # self.use_html = opt.isTrain and not opt.no_html
        # self.win_size = opt.display_winsize
        self.name = opt.name
        self.mode = opt.mode
        self.real_train = opt.real_train if opt.isTrain else False
        if opt.isTrain:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        for label, image_numpy in visuals.items():
            # print('label',label)
            # print('image_numpy',image_numpy.shape)
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%d_%s_%d.jpg' % (epoch, step, label, i))
                    util.save_image(image_numpy[i], img_path)
            else:
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%d_%s.jpg' % (epoch, step, label))
                util.save_image(image_numpy, img_path)

        # if self.use_html: # save images to a html file
        #     webpage = html.HTML(self.web_dir, self.mode, 'Experiment name = %s' % self.name, refresh=30)
        #     for n in range(epoch, 0, -1):
        #         webpage.add_header('epoch [%d]' % n)
        #         ims = []
        #         txts = []
        #         links = []

        #         for label, image_numpy in visuals.items():
        #             if isinstance(image_numpy, list):
        #                 for i in range(len(image_numpy)):
        #                     img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
        #                     ims.append(img_path)
        #                     txts.append(label+str(i))
        #                     links.append(img_path)
        #             else:
        #                 img_path = 'epoch%.3d_%s.jpg' % (n, label)
        #                 ims.append(img_path)
        #                 txts.append(label)
        #                 links.append(img_path)
        #         if len(ims) < 10:
        #             webpage.add_images(ims, txts, links, width=self.win_size)
        #         else:
        #             num = int(round(len(ims)/2.0))
        #             webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
        #             webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
        #     webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()

        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        
        if self.real_train:
            nametag=os.path.split(os.path.split(image_path[0])[0])[1]

        # webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            if self.real_train:
                image_name = '%s_%s_%s.jpg' % (nametag, name, label)
            else:
                image_name = '%s_%s.jpg' % (name, label)

            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)
            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        # webpage.add_images(ims, txts, links, width=self.win_size)
