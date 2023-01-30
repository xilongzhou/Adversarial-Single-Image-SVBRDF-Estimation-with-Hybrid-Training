import time
import os
from os.path import join

import numpy as np
import torch

from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

import random
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter 

import matplotlib.pyplot as plt
from util.util import *


if __name__ == '__main__':

	myseed = random.randint(0, 2**31 - 1)
	torch.manual_seed(myseed)
	torch.cuda.manual_seed(myseed)
	torch.cuda.manual_seed_all(myseed) 
	np.random.seed(myseed)

	opt = TrainOptions().parse(myseed)

	iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
	if opt.continue_train:
		try:
			print(iter_path)
			if opt.which_epoch=='latest':				
				start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
				print(np.loadtxt(iter_path , delimiter=',', dtype=int))
				print('start_epoch:', start_epoch)
			else:
				start_epoch = int(opt.which_epoch)+1
				epoch_iter = 0 
		except:
			start_epoch, epoch_iter = 1, 0

		print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
	else:    
		start_epoch, epoch_iter = 1, 0

	opt.print_freq = lcm(opt.print_freq, opt.batchSize)    

	## gamma correction for display
	Gamma_Correciton=False
	if opt.MyTest == 'Diff' or opt.MyTest == 'Spec':
		Gamma_Correciton=True
	print('Gamma Correction: ',Gamma_Correciton)

	data_loader = CreateDataLoader(opt)
	if opt.real_train:
		dataset, real_dataset = data_loader.load_data()
	else:
		dataset = data_loader.load_data()
	dataset_size = len(data_loader)

	print('#training images = %d' % dataset_size)

	model = create_model(opt)
	visualizer = Visualizer(opt)

	if opt.MyTest == 'ALL_4D':
		optimizer_G = model.module.optimizer_G
		optimizer_D_norm = model.module.optimizer_D_Norm
		optimizer_D_diff = model.module.optimizer_D_Diff
		optimizer_D_rough = model.module.optimizer_D_Rough
		optimizer_D_spec = model.module.optimizer_D_Spec
	elif opt.MyTest=='ALL_5D_Render':
		optimizer_G = model.module.optimizer_G
		optimizer_D_norm = model.module.optimizer_D_Norm
		optimizer_D_diff = model.module.optimizer_D_Diff
		optimizer_D_rough = model.module.optimizer_D_Rough
		optimizer_D_spec = model.module.optimizer_D_Spec
		optimizer_D_render = model.module.optimizer_D_Render
	elif opt.MyTest=='ALL_1D_Render':
		optimizer_G, optimizer_D_render = model.module.optimizer_G, model.module.optimizer_D_Render
	elif opt.MyTest=='L1' or opt.MyTest=='L1_Render':
		optimizer_G = model.module.optimizer_G

	if opt.real_train:
		optimizer_D_render_real = model.module.optimizer_D_Render_Real

	total_steps = (start_epoch-1) * dataset_size + epoch_iter

	display_delta = total_steps % opt.display_freq
	print_delta = total_steps % opt.print_freq
	save_delta = total_steps % opt.save_latest_freq

	#################### logs or not ########################
	if opt.logs:
		print('start logging:')
		log_save_direct = opt.savelog_dir
		if not os.path.exists(log_save_direct):         
			os.makedirs(log_save_direct)
		if opt.continue_train:
			writer = SummaryWriter(join(log_save_direct,'{}_resume'.format(opt.name)),flush_secs=10)
		else:
			writer = SummaryWriter(join(log_save_direct,'{}'.format(opt.name)),flush_secs=10)
	else:
		print('not logging')

	print(start_epoch)
	for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		if epoch != start_epoch:
			epoch_iter = epoch_iter % dataset_size

		for i, data in enumerate(dataset, start=epoch_iter):
			
			############################################## Loading Input Data ################################################
			# loading real images 
			real_data_pair = torch.zeros([1], device='cuda')
			if opt.real_train:
				global RealTrainData
				try:
					# print('try')
					real_data = RealTrainData.next()
				except NameError:
					# print('name error')
					RealTrainData = iter(real_dataset)
					real_data = RealTrainData.next()
				except StopIteration:
					# print('StopIteration')
					RealTrainData = iter(real_dataset)
					real_data = RealTrainData.next()

				# concat syn and real data
				real_data_pair = torch.cat((real_data['label'],real_data['image']),0)

			if total_steps % opt.print_freq == print_delta:
				iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize

			# whether to collect output images
			save_fake = total_steps % opt.display_freq == display_delta
			
			############## Forward Pass ######################
			losses, generated, rendered, inputimage, light, real, real_l1loss = model(data['label'],data['image'], real_data_pair, infer=save_fake)

			losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]

			loss_dict = dict(zip(model.module.loss_names, losses))

			# calculate final loss scalar
			if opt.MyTest=='ALL_4D':
				loss_D_norm = (loss_dict['D_fake_norm'] + loss_dict['D_real_norm']) * 0.5
				loss_D_diff = (loss_dict['D_fake_diff'] + loss_dict['D_real_diff']) * 0.5
				loss_D_rough = (loss_dict['D_fake_rough'] + loss_dict['D_real_rough']) * 0.5
				loss_D_spec = (loss_dict['D_fake_spec'] + loss_dict['D_real_spec']) * 0.5
			elif opt.MyTest=='ALL_5D_Render':
				loss_D_norm = (loss_dict['D_fake_norm'] + loss_dict['D_real_norm']) * 0.5
				loss_D_diff = (loss_dict['D_fake_diff'] + loss_dict['D_real_diff']) * 0.5
				loss_D_rough = (loss_dict['D_fake_rough'] + loss_dict['D_real_rough']) * 0.5
				loss_D_spec = (loss_dict['D_fake_spec'] + loss_dict['D_real_spec']) * 0.5				
				loss_D_render = (loss_dict['D_fake_render'] + loss_dict['D_real_render']) * 0.5	
			elif opt.MyTest=='ALL_1D_Render':
				loss_D_render = (loss_dict['D_fake_render'] + loss_dict['D_real_render']) * 0.5	

			if opt.real_train:
				loss_D_render_real = (loss_dict['realD_fake'] + loss_dict['realD_real']) * 0.5

			loss_G = loss_dict.get('G_GAN',0) + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_L1',0) + loss_dict.get('light_l1',0)

			############### Backward Pass ####################
			# update generator weights
			optimizer_G.zero_grad()
			loss_G.backward()          
			optimizer_G.step()

			# update discriminator weights
			if opt.MyTest == 'ALL_4D':
				optimizer_D_norm.zero_grad()
				loss_D_norm.backward()        
				optimizer_D_norm.step() 
			
				optimizer_D_diff.zero_grad()
				loss_D_diff.backward()        
				optimizer_D_diff.step() 

				optimizer_D_rough.zero_grad()
				loss_D_rough.backward()        
				optimizer_D_rough.step() 

				optimizer_D_spec.zero_grad()
				loss_D_spec.backward()        
				optimizer_D_spec.step() 
			elif opt.MyTest=='ALL_5D_Render':
				optimizer_D_norm.zero_grad()
				loss_D_norm.backward()        
				optimizer_D_norm.step() 
			
				optimizer_D_diff.zero_grad()
				loss_D_diff.backward()        
				optimizer_D_diff.step() 

				optimizer_D_rough.zero_grad()
				loss_D_rough.backward()        
				optimizer_D_rough.step() 

				optimizer_D_spec.zero_grad()
				loss_D_spec.backward()        
				optimizer_D_spec.step() 

				optimizer_D_render.zero_grad()
				loss_D_render.backward()        
				optimizer_D_render.step() 								
			elif opt.MyTest=='ALL_1D_Render':				
				optimizer_D_render.zero_grad()
				loss_D_render.backward()        
				optimizer_D_render.step()        


			if opt.real_train:
				optimizer_D_render_real.zero_grad()
				loss_D_render_real.backward()        
				optimizer_D_render_real.step() 				

			############## Display results and errors ##########
			### print out errors
			if total_steps % opt.print_freq == print_delta:
				if light is not None:
					print('fake light', light['Fake'], 'real light', light['Real'])

				errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}  
				if opt.real_train:
					real_l1={'RealL1':real_l1loss.data.item()}
					errors.update(real_l1)

				t = (time.time() - iter_start_time) / opt.print_freq
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				# visualizer.plot_current_errors(errors, total_steps)
				if opt.logs:
					for tag, value in errors.items():
						writer.add_scalar(tag, value, total_steps)
				#call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

			### display output images
			if save_fake:
				if opt.MyTest=='ALL_1D_Render' or opt.MyTest=='ALL_5D_Render' or opt.MyTest=='L1_Render':
					visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
										   ('Normal Fake', util.tensor2im(generated.data[0,0:3,:,:], gamma=False)),
										   ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
										   ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
										   ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
										   ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False)),
										   ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
										   ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True)),
										   ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True)),
										   ('Render Fake', util.tensor2im(rendered['Fake'].data[0,:,:,:], gamma=True)),
										   ('Render Real', util.tensor2im(rendered['Real'].data[0,:,:,:], gamma=True))])

				elif opt.MyTest=='L1' or opt.MyTest=='ALL_4D':
					visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
										   ('Normal Fake', util.tensor2im(generated.data[0,0:3,:,:], gamma=False)),
										   ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
										   ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
										   ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
										   ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False)),
										   ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
										   ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True)),
										   ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True))])
				elif opt.MyTest=='Diff':
					visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
										   ('synthesized_image', util.tensor2im(generated.data[0], gamma=True)),
										   ('real_image', util.tensor2im(data['image'][0,3:6,:,:], gamma=True))])
				else:
					visuals = OrderedDict([('input_label', util.tensor2im(data['label'][0], gamma=False)),
										   ('synthesized_image', util.tensor2im(generated.data[0], gamma=Gamma_Correciton)),
										   ('real_image', util.tensor2im(data['image'][0], gamma=Gamma_Correciton))])

				if opt.real_train:
					print('fake light, ', real['fakeLightB'].data[0])

					visual_temp = OrderedDict([('RealTrain_A', util.tensor2im(real['image_A'][0], gamma=False)),
										   ('RealTrain_B', util.tensor2im(real['image_B'][0], gamma=False)),
										   ('RealTrain_B_fake', util.tensor2im(real['fake_B'].data[0], gamma=False)),
										   ('RealTrain_B_fakebefore', util.tensor2im(real['fake_B_before'].data[0], gamma=False)),
										   ('RealTrain_A_Diff', util.tensor2im(real['Diff_A'].data[0], gamma=True)),
										   ('RealTrain_A_Norm', util.tensor2im(real['Norm_A'].data[0], gamma=False)),
										   ('RealTrain_A_Rough', util.tensor2im(real['Rough_A'].data[0], gamma=False)),
										   ('RealTrain_A_Spec', util.tensor2im(real['Spec_A'].data[0], gamma=True)),
										   ('RealTrain_B_Diff', util.tensor2im(real['Diff_B'].data[0], gamma=True)),
										   ('RealTrain_B_Norm', util.tensor2im(real['Norm_B'].data[0], gamma=False)),
										   ('RealTrain_B_Rough', util.tensor2im(real['Rough_B'].data[0], gamma=False)),
										   ('RealTrain_B_Spec', util.tensor2im(real['Spec_B'].data[0], gamma=True))
										   ])

					visuals.update(visual_temp)
					# print(visuals)

				visualizer.display_current_results(visuals, epoch, total_steps)
				if opt.logs:
					for tag, images in visuals.items():
						writer.add_image(tag, np.transpose(images, (2, 0, 1)), total_steps)


			### save latest model
			if total_steps % opt.save_latest_freq == save_delta:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				# model.module.save('latest')            
				model.module.save(str(epoch-1)+'_'+str(epoch_iter))            
				np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

			if epoch_iter >= dataset_size:
				break
		   
		# end of epoch 
		iter_end_time = time.time()
		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		### save model for this epoch
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
			model.module.save('latest')
			model.module.save(epoch)
			np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
			
		### linearly decay learning rate after certain iterations
		# if epoch > opt.niter:
		# 	model.module.update_learning_rate()
