import os
from collections import OrderedDict

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.renderer import *

import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import ntpath

if __name__ == '__main__':

	opt = TestOptions().parse(seed=0,save=False)
	opt.nThreads = 1   # test code only supports nThreads = 1
	opt.batchSize = 1  # test code only supports batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip

	myseed = 0 #random.randint(0, 2**31 - 1)
	torch.manual_seed(myseed)
	torch.cuda.manual_seed(myseed)
	torch.cuda.manual_seed_all(myseed) 
	np.random.seed(myseed)

	# load data
	data_loader = CreateDataLoader(opt) 
	mydata = data_loader.load_data()
	
	# saving path	
	visualizer = Visualizer(opt)
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir,opt.savename, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

	# if save light pos
	if opt.savelight_to_one:
		txtpath_li=os.path.join(web_dir,'light_{}'.format(opt.savename))
		if not os.path.exists(txtpath_li):
			os.makedirs(txtpath_li)    
		txtfile_li=open(os.path.join(txtpath_li,'Light.txt'),'w')
		txtfile_li.write('predefined N: {}\n'.format(len(mydata)))

	# load model
	model = create_model(opt)
	model.eval()

	for i, data in enumerate(mydata):

		img_path = data['path']
		print('process image... %s' % img_path)

		if opt.MyTest=='ALL_5D_Render':
			generated,_, rendered,inputimage,fakelight = model.inference(data['label'], data['image'])
		else: 
			generated,_,inputimage,fakelight = model.inference(data['label'], data['image'])
			 

		# output light position 
		if opt.savelight_to_one:
			txtfile_li.write('{:.6f},{:.6f},{:.6f}\t'.format(fakelight[0,0].cpu().numpy(),fakelight[0,1].cpu().numpy(),fakelight[0,2].cpu().numpy()))
			txtfile_li.write('{:.6f},{:.6f},{:.6f}\n'.format(0.0,0.0,2.14))
		elif opt.savelight_to_multi:
			short_path = ntpath.basename(img_path[0])
			name = os.path.splitext(short_path)[0]
			txtpath_li=os.path.join(web_dir,'light_{}'.format(opt.savename))
			if not os.path.exists(txtpath_li):
				os.makedirs(txtpath_li)                

			txtfile_li=open(os.path.join(txtpath_li,'{}.txt'.format(name[:-4] if len(name)>4 else name)),'w')
			txtfile_li.write('{}\n'.format(name[:-4]))
			txtfile_li.write('{:.6f},{:.6f},{:.6f}\t'.format(fakelight[0,0].cpu().numpy(),fakelight[0,1].cpu().numpy(),fakelight[0,2].cpu().numpy()))
			txtfile_li.write('{:.6f},{:.6f},{:.6f}\n'.format(0.0,0.0,2.14))
			txtfile_li.close()  

		# output images
		if opt.mode =='Syn':


			if opt.MyTest=='ALL_5D_Render':
				normal_image = generated.data[0,0:3,:,:].detach()*2-1
				Normal_vec = (normalize_vec(normal_image.permute(1,2,0))+1)*0.5
				visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
									 ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False, normalize=False)),
									 ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
									 ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True, normalize=False)),
									 ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
									 ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False, normalize=False)),
									 ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
									 ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True, normalize=False)),
									 ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True)),
									 ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True, normalize=False))
									 # ('Render Real', util.tensor2im(rendered['Real'][0,:,:,:], gamma=True))
									 ])
			elif opt.MyTest=='ALL_4D':
				Normal_vec = normalize_vec(generated.data[0,0:3,:,:].detach().permute(1,2,0))
				visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
									 ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False)),
									 ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
									 ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
									 ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
									 ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False)),
									 ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
									 ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True)),
									 ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True))])		
		elif opt.mode =='Real':
			if opt.MyTest=='ALL_5D_Render':
				# process normal
				normal_image = generated.data[0,0:3,:,:].detach()*2-1
				Normal_vec = (normalize_vec(normal_image.permute(1,2,0))+1)*0.5
				visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
															 ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False, normalize=False)),
															 ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True, normalize=False)),
															 ('Rough Fake', util.tensor2im(generated.data[0,8:9,:,:].repeat(3,1,1), gamma=False, normalize=False)),
															 ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True, normalize=False)),
															 ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True,normalize=False))
															 ])            
			else:
				# process normal
				Normal_vec = normalize_vec(generated.data[0,0:3,:,:].detach().permute(1,2,0))
				visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
															 ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False)),
															 ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
															 ('Rough Fake', util.tensor2im(generated.data[0,8:9,:,:].repeat(3,1,1), gamma=False)),
															 ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True))
															 # ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True))
															 ]) 

		visualizer.save_images(webpage, visuals, img_path)

	if opt.savelight_to_one:
		txtfile_li.close()

