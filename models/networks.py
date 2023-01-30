import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, 0.01*y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.002) 


def VA_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('InstanceNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)    
	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, 0.01*y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.002) 


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def define_G(input_nc, output_nc,rough_channel, netG, gpu_ids=[]):

	# if netG == 'global':    
	# 	netG = GlobalGenerator(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, norm_layer)
	# elif netG == 'global_Light':    
	# 	netG = GlobalGenerator_Light(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, norm_layer)
	# elif netG == 'newarch':    
	# 	netG = NewGenerator(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, n_blocks_branch, norm_layer)   
	# elif netG == 'newarch_Light':
	# 	netG = NewGenerator_Light(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, n_blocks_branch, norm_layer)   
	# elif netG == 'local':        
	# 	netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,n_local_enhancers, n_blocks_local, norm_layer)
	# elif netG == 'encoder':
	# 	netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
	# elif netG == 'VA_Net':
	# 	netG = VA_Net(input_nc,output_nc)
	# elif netG =='LocalVA_Net':
	# 	netG = LocalVA_Net(input_nc,output_nc)
	if netG =='NewVA_Net':
		netG = NewVA_Net(input_nc,output_nc,rough_channel)
	elif netG =='NewVA_Net_Light':
		netG = NewVA_Net_Light(input_nc,output_nc,rough_channel)
	else:
		raise('generator not implemented!')
	print(netG)

	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())   
		netG.cuda(gpu_ids[0])

	print('VA Net initialization')
	netG.apply(VA_weights_init)
	# else:
	# 	print('Other Net initialization')
	# 	netG.apply(weights_init)

	return netG

def define_D(input_nc, dropout, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
	norm_layer = get_norm_layer(norm_type=norm)   
	netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,dropout)   
	print(netD)
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		netD.cuda(gpu_ids[0])
	netD.apply(weights_init)
	return netD

def print_network(net):
	if isinstance(net, list):
		net = net[0]
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

# ------------------------------------------ Losses --------------------------------------------

class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		if isinstance(input[0], list):
			loss = 0
			for input_i in input:
				pred = input_i[-1]
				target_tensor = self.get_target_tensor(pred, target_is_real)
				loss += self.loss(pred, target_tensor)
			return loss
		else:            
			target_tensor = self.get_target_tensor(input[-1], target_is_real)
			return self.loss(input[-1], target_tensor)

class GANLoss_Smooth(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=0.9, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss_Smooth, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		if isinstance(input[0], list):
			loss = 0
			for input_i in input:
				pred = input_i[-1]
				target_tensor = self.get_target_tensor(pred, target_is_real)
				loss += self.loss(pred, target_tensor)
			return loss
		else:            
			target_tensor = self.get_target_tensor(input[-1], target_is_real)
			return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
	def __init__(self, gpu_ids):
		super(VGGLoss, self).__init__()        
		self.vgg = Vgg19().cuda()
		self.criterion = nn.L1Loss()
		self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

	def forward(self, x, y):              
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
		return loss


# ------------------------------------- Generator architecture ----------------------------------


class FC(nn.Module):
	def __init__(self,input_channel,output_channel,BIAS):
		super(FC,self).__init__()
		self.fc_layer=nn.Linear(input_channel,output_channel,bias=BIAS)

	def forward(self,input):
		# if input is 4D [b,c,1,1],output [b,c,1,1]
		if len(input.shape) == 4:
			[b,c,w,h]=input.shape
			out=self.fc_layer(input.view(b,c))
			out=out.unsqueeze(2).unsqueeze(2)
		# if input is 2D [b,c] otuput [b,c]
		elif len(input.shape) == 2:
			out=self.fc_layer(input)
		# otherwise, error
		else:
			print('incorrectly input to FC layer')
			sys.exit(1) 

		return out


class Deconv(nn.Module):

	def __init__(self,input_channel,output_channel):

		super(Deconv,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		# self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')

		self.temp_conv1 = nn.Conv2d(input_channel,output_channel,4,stride=1,bias=False)
		self.temp_conv2 = nn.Conv2d(output_channel,output_channel,4,stride=1,bias=False)

		# realize same padding in tensorflow
		self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)

	def forward(self,input):

		# print('Deco input shape,',input.shape[1])
		# Upsamp=self.upsampling(input)
		## hack upsampling method to make is deterministic
		Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)

		out=self.temp_conv1(self.padding(Upsamp))
		out=self.temp_conv2(self.padding(out))

		# print('output shape,',out.shape)
		return out

def mymean(input):
	[b,c,w,h]=input.shape
	mean=input.view(b,c,-1).mean(2)
	mean=mean.unsqueeze(-1).unsqueeze(-1)
	return mean#.reshape(b,c,1,1)


class NewVA_Net(nn.Module):
	def __init__(self,input_channel,output_channel, rough_channel):
		super(NewVA_Net,self).__init__()
		
		self.rough_nc=rough_channel

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder(diff)
		self.deconv1_diff = Deconv(512, 512)
		self.deconv2_diff = Deconv(1024, 512)
		self.deconv3_diff = Deconv(1024, 512)
		self.deconv4_diff = Deconv(1024, 512)
		self.deconv5_diff = Deconv(1024, 256)
		self.deconv6_diff = Deconv(512, 128)
		self.deconv7_diff = Deconv(256, 64)
		self.deconv8_diff = Deconv(128, output_channel)

		#decoder(normal)
		self.deconv1_normal = Deconv(512, 512)
		self.deconv2_normal = Deconv(1024, 512)
		self.deconv3_normal = Deconv(1024, 512)
		self.deconv4_normal = Deconv(1024, 512)
		self.deconv5_normal = Deconv(1024, 256)
		self.deconv6_normal = Deconv(512, 128)
		self.deconv7_normal = Deconv(256, 64)
		self.deconv8_normal = Deconv(128, output_channel)

		#decoder(rough)
		self.deconv1_rough = Deconv(512, 512)
		self.deconv2_rough = Deconv(1024, 512)
		self.deconv3_rough = Deconv(1024, 512)
		self.deconv4_rough = Deconv(1024, 512)
		self.deconv5_rough = Deconv(1024, 256)
		self.deconv6_rough = Deconv(512, 128)
		self.deconv7_rough = Deconv(256, 64)
		self.deconv8_rough = Deconv(128, rough_channel)

		#decoder(spec)
		self.deconv1_spec = Deconv(512, 512)
		self.deconv2_spec = Deconv(1024, 512)
		self.deconv3_spec = Deconv(1024, 512)
		self.deconv4_spec = Deconv(1024, 512)
		self.deconv5_spec = Deconv(1024, 256)
		self.deconv6_spec = Deconv(512, 128)
		self.deconv7_spec = Deconv(256, 64)
		self.deconv8_spec = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_diff = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_diff = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_diff = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_normal = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_normal = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_normal = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_rough = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_rough = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_rough = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_spec = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_spec = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_spec = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)


	def forward(self, input):

		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network
		# [batch,128,h/4,w/4]        
		encoder2 = self.instance_normal2(self.conv2(self.leaky_relu(encoder1))) #local network
		# [batch,256,h/8,w/8]        
		encoder3 = self.instance_normal3(self.conv3(self.leaky_relu(encoder2))) #local network
		# [batch,512,h/16,w/16]        
		encoder4 = self.instance_normal4(self.conv4(self.leaky_relu(encoder3))) #local network
		# [batch,512,h/32,w/32]        
		encoder5 = self.instance_normal5(self.conv5(self.leaky_relu(encoder4))) #local network
		# [batch,512,h/64,w/64]        
		encoder6 = self.instance_normal6(self.conv6(self.leaky_relu(encoder5))) #local network
		# [batch,512,h/128,w/128]        
		encoder7 = self.instance_normal7(self.conv7(self.leaky_relu(encoder6))) #local network
		# [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder (diff) #############################################
		# [batch,512,h/128,w/128]
		decoder1_diff = self.instance_normal_de_1_diff(self.deconv1_diff(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_diff = torch.cat((self.dropout(decoder1_diff), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_diff = self.instance_normal_de_2_diff(self.deconv2_diff(self.leaky_relu(decoder1_diff)))
		# [batch,1024,h/64,w/64]
		decoder2_diff = torch.cat((self.dropout(decoder2_diff), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_diff = self.instance_normal_de_3_diff(self.deconv3_diff(self.leaky_relu(decoder2_diff)))
		# [batch,1024,h/32,w/32]
		decoder3_diff = torch.cat((self.dropout(decoder3_diff), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_diff = self.instance_normal_de_4_diff(self.deconv4_diff(self.leaky_relu(decoder3_diff)))
		# [batch,1024,h/16,w/16]
		decoder4_diff = torch.cat((decoder4_diff, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_diff = self.instance_normal_de_5_diff(self.deconv5_diff(self.leaky_relu(decoder4_diff)))
		# [batch,512,h/8,w/8]
		decoder5_diff = torch.cat((decoder5_diff, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_diff = self.instance_normal_de_6_diff(self.deconv6_diff(self.leaky_relu(decoder5_diff)))
		# [batch,256,h/4,w/4]
		decoder6_diff = torch.cat((decoder6_diff, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_diff = self.instance_normal_de_7_diff(self.deconv7_diff(self.leaky_relu(decoder6_diff)))
		# [batch,128,h/2,w/2]
		decoder7_diff = torch.cat((decoder7_diff, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_diff = self.deconv8_diff(self.leaky_relu(decoder7_diff))

		diff = self.tan(decoder8_diff)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_normal = self.instance_normal_de_1_normal(self.deconv1_normal(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_normal = torch.cat((self.dropout(decoder1_normal), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_normal = self.instance_normal_de_2_normal(self.deconv2_normal(self.leaky_relu(decoder1_normal)))
		# [batch,1024,h/64,w/64]
		decoder2_normal = torch.cat((self.dropout(decoder2_normal), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_normal = self.instance_normal_de_3_normal(self.deconv3_normal(self.leaky_relu(decoder2_normal)))
		# [batch,1024,h/32,w/32]
		decoder3_normal = torch.cat((self.dropout(decoder3_normal), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_normal = self.instance_normal_de_4_normal(self.deconv4_normal(self.leaky_relu(decoder3_normal)))
		# [batch,1024,h/16,w/16]
		decoder4_normal = torch.cat((decoder4_normal, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_normal = self.instance_normal_de_5_normal(self.deconv5_normal(self.leaky_relu(decoder4_normal)))
		# [batch,512,h/8,w/8]
		decoder5_normal = torch.cat((decoder5_normal, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_normal = self.instance_normal_de_6_normal(self.deconv6_normal(self.leaky_relu(decoder5_normal)))
		# [batch,256,h/4,w/4]
		decoder6_normal = torch.cat((decoder6_normal, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_normal = self.instance_normal_de_7_normal(self.deconv7_normal(self.leaky_relu(decoder6_normal)))
		# [batch,128,h/2,w/2]
		decoder7_normal = torch.cat((decoder7_normal, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_normal = self.deconv8_normal(self.leaky_relu(decoder7_normal))

		normal = self.tan(decoder8_normal)
		# print(output.shape)
	 
		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_rough = self.instance_normal_de_1_rough(self.deconv1_rough(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_rough = torch.cat((self.dropout(decoder1_rough), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_rough = self.instance_normal_de_2_rough(self.deconv2_rough(self.leaky_relu(decoder1_rough)))
		# [batch,1024,h/64,w/64]
		decoder2_rough = torch.cat((self.dropout(decoder2_rough), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_rough = self.instance_normal_de_3_rough(self.deconv3_rough(self.leaky_relu(decoder2_rough)))
		# [batch,1024,h/32,w/32]
		decoder3_rough = torch.cat((self.dropout(decoder3_rough), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_rough = self.instance_normal_de_4_rough(self.deconv4_rough(self.leaky_relu(decoder3_rough)))
		# [batch,1024,h/16,w/16]
		decoder4_rough = torch.cat((decoder4_rough, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_rough = self.instance_normal_de_5_rough(self.deconv5_rough(self.leaky_relu(decoder4_rough)))
		# [batch,512,h/8,w/8]
		decoder5_rough = torch.cat((decoder5_rough, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_rough = self.instance_normal_de_6_rough(self.deconv6_rough(self.leaky_relu(decoder5_rough)))
		# [batch,256,h/4,w/4]
		decoder6_rough = torch.cat((decoder6_rough, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_rough = self.instance_normal_de_7_rough(self.deconv7_rough(self.leaky_relu(decoder6_rough)))
		# [batch,128,h/2,w/2]
		decoder7_rough = torch.cat((decoder7_rough, encoder1), 1)

		# [batch,_out_c,h,w]
		decoder8_rough = self.deconv8_rough(self.leaky_relu(decoder7_rough))

		rough = self.tan(decoder8_rough)
		# print(output.shape)
		if self.rough_nc==1:
			rough=rough.repeat(1,3,1,1)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_spec = self.instance_normal_de_1_spec(self.deconv1_spec(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_spec = torch.cat((self.dropout(decoder1_spec), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_spec = self.instance_normal_de_2_spec(self.deconv2_spec(self.leaky_relu(decoder1_spec)))
		# [batch,1024,h/64,w/64]
		decoder2_spec = torch.cat((self.dropout(decoder2_spec), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_spec = self.instance_normal_de_3_spec(self.deconv3_spec(self.leaky_relu(decoder2_spec)))
		# [batch,1024,h/32,w/32]
		decoder3_spec = torch.cat((self.dropout(decoder3_spec), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_spec = self.instance_normal_de_4_spec(self.deconv4_spec(self.leaky_relu(decoder3_spec)))
		# [batch,1024,h/16,w/16]
		decoder4_spec = torch.cat((decoder4_spec, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_spec = self.instance_normal_de_5_spec(self.deconv5_spec(self.leaky_relu(decoder4_spec)))
		# [batch,512,h/8,w/8]
		decoder5_spec = torch.cat((decoder5_spec, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_spec = self.instance_normal_de_6_spec(self.deconv6_spec(self.leaky_relu(decoder5_spec)))
		# [batch,256,h/4,w/4]
		decoder6_spec = torch.cat((decoder6_spec, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_spec = self.instance_normal_de_7_spec(self.deconv7_spec(self.leaky_relu(decoder6_spec)))
		# [batch,128,h/2,w/2]
		decoder7_spec = torch.cat((decoder7_spec, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_spec = self.deconv8_spec(self.leaky_relu(decoder7_spec))

		spec = self.tan(decoder8_spec)

		output=torch.cat((normal,diff,rough,spec),1)

		# print('shape: ',output.shape)

		return output, None


class NewVA_Net_Light(nn.Module):
	def __init__(self,input_channel,output_channel,rough_channel):
		super(NewVA_Net_Light,self).__init__()

		self.rough_nc=rough_channel
		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder(diff)
		self.deconv1_diff = Deconv(512, 512)
		self.deconv2_diff = Deconv(1024, 512)
		self.deconv3_diff = Deconv(1024, 512)
		self.deconv4_diff = Deconv(1024, 512)
		self.deconv5_diff = Deconv(1024, 256)
		self.deconv6_diff = Deconv(512, 128)
		self.deconv7_diff = Deconv(256, 64)
		self.deconv8_diff = Deconv(128, output_channel)

		#decoder(normal)
		self.deconv1_normal = Deconv(512, 512)
		self.deconv2_normal = Deconv(1024, 512)
		self.deconv3_normal = Deconv(1024, 512)
		self.deconv4_normal = Deconv(1024, 512)
		self.deconv5_normal = Deconv(1024, 256)
		self.deconv6_normal = Deconv(512, 128)
		self.deconv7_normal = Deconv(256, 64)
		self.deconv8_normal = Deconv(128, output_channel)

		#decoder(rough)
		self.deconv1_rough = Deconv(512, 512)
		self.deconv2_rough = Deconv(1024, 512)
		self.deconv3_rough = Deconv(1024, 512)
		self.deconv4_rough = Deconv(1024, 512)
		self.deconv5_rough = Deconv(1024, 256)
		self.deconv6_rough = Deconv(512, 128)
		self.deconv7_rough = Deconv(256, 64)
		self.deconv8_rough = Deconv(128, rough_channel)

		#decoder(spec)
		self.deconv1_spec = Deconv(512, 512)
		self.deconv2_spec = Deconv(1024, 512)
		self.deconv3_spec = Deconv(1024, 512)
		self.deconv4_spec = Deconv(1024, 512)
		self.deconv5_spec = Deconv(1024, 256)
		self.deconv6_spec = Deconv(512, 128)
		self.deconv7_spec = Deconv(256, 64)
		self.deconv8_spec = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_diff = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_diff = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_diff = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_normal = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_normal = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_normal = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_rough = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_rough = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_rough = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_spec = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_spec = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_spec = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)

		########################### This is for light ##########################
		# self.Mean=mymean()
		self.FC1_Light=FC(512*1*1,256,True)
		self.FC2_Light=FC(256,128,True)
		self.FC3_Light=FC(128,64,True)
		self.FC4_Light=FC(64,32,True)
		self.FC5_Light=FC(32,16,True)
		self.FC6_Light=FC(16,3,True)


	def forward(self, input):

		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network
		# [batch,128,h/4,w/4]        
		encoder2 = self.instance_normal2(self.conv2(self.leaky_relu(encoder1))) #local network
		# [batch,256,h/8,w/8]        
		encoder3 = self.instance_normal3(self.conv3(self.leaky_relu(encoder2))) #local network
		# [batch,512,h/16,w/16]        
		encoder4 = self.instance_normal4(self.conv4(self.leaky_relu(encoder3))) #local network
		# [batch,512,h/32,w/32]        
		encoder5 = self.instance_normal5(self.conv5(self.leaky_relu(encoder4))) #local network
		# [batch,512,h/64,w/64]        
		encoder6 = self.instance_normal6(self.conv6(self.leaky_relu(encoder5))) #local network
		# [batch,512,h/128,w/128]        
		encoder7 = self.instance_normal7(self.conv7(self.leaky_relu(encoder6))) #local network
		# [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder (diff) #############################################
		# [batch,512,h/128,w/128]
		decoder1_diff = self.instance_normal_de_1_diff(self.deconv1_diff(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_diff = torch.cat((self.dropout(decoder1_diff), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_diff = self.instance_normal_de_2_diff(self.deconv2_diff(self.leaky_relu(decoder1_diff)))
		# [batch,1024,h/64,w/64]
		decoder2_diff = torch.cat((self.dropout(decoder2_diff), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_diff = self.instance_normal_de_3_diff(self.deconv3_diff(self.leaky_relu(decoder2_diff)))
		# [batch,1024,h/32,w/32]
		decoder3_diff = torch.cat((self.dropout(decoder3_diff), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_diff = self.instance_normal_de_4_diff(self.deconv4_diff(self.leaky_relu(decoder3_diff)))
		# [batch,1024,h/16,w/16]
		decoder4_diff = torch.cat((decoder4_diff, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_diff = self.instance_normal_de_5_diff(self.deconv5_diff(self.leaky_relu(decoder4_diff)))
		# [batch,512,h/8,w/8]
		decoder5_diff = torch.cat((decoder5_diff, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_diff = self.instance_normal_de_6_diff(self.deconv6_diff(self.leaky_relu(decoder5_diff)))
		# [batch,256,h/4,w/4]
		decoder6_diff = torch.cat((decoder6_diff, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_diff = self.instance_normal_de_7_diff(self.deconv7_diff(self.leaky_relu(decoder6_diff)))
		# [batch,128,h/2,w/2]
		decoder7_diff = torch.cat((decoder7_diff, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_diff = self.deconv8_diff(self.leaky_relu(decoder7_diff))

		diff = self.tan(decoder8_diff)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_normal = self.instance_normal_de_1_normal(self.deconv1_normal(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_normal = torch.cat((self.dropout(decoder1_normal), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_normal = self.instance_normal_de_2_normal(self.deconv2_normal(self.leaky_relu(decoder1_normal)))
		# [batch,1024,h/64,w/64]
		decoder2_normal = torch.cat((self.dropout(decoder2_normal), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_normal = self.instance_normal_de_3_normal(self.deconv3_normal(self.leaky_relu(decoder2_normal)))
		# [batch,1024,h/32,w/32]
		decoder3_normal = torch.cat((self.dropout(decoder3_normal), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_normal = self.instance_normal_de_4_normal(self.deconv4_normal(self.leaky_relu(decoder3_normal)))
		# [batch,1024,h/16,w/16]
		decoder4_normal = torch.cat((decoder4_normal, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_normal = self.instance_normal_de_5_normal(self.deconv5_normal(self.leaky_relu(decoder4_normal)))
		# [batch,512,h/8,w/8]
		decoder5_normal = torch.cat((decoder5_normal, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_normal = self.instance_normal_de_6_normal(self.deconv6_normal(self.leaky_relu(decoder5_normal)))
		# [batch,256,h/4,w/4]
		decoder6_normal = torch.cat((decoder6_normal, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_normal = self.instance_normal_de_7_normal(self.deconv7_normal(self.leaky_relu(decoder6_normal)))
		# [batch,128,h/2,w/2]
		decoder7_normal = torch.cat((decoder7_normal, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_normal = self.deconv8_normal(self.leaky_relu(decoder7_normal))

		normal = self.tan(decoder8_normal)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_rough = self.instance_normal_de_1_rough(self.deconv1_rough(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_rough = torch.cat((self.dropout(decoder1_rough), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_rough = self.instance_normal_de_2_rough(self.deconv2_rough(self.leaky_relu(decoder1_rough)))
		# [batch,1024,h/64,w/64]
		decoder2_rough = torch.cat((self.dropout(decoder2_rough), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_rough = self.instance_normal_de_3_rough(self.deconv3_rough(self.leaky_relu(decoder2_rough)))
		# [batch,1024,h/32,w/32]
		decoder3_rough = torch.cat((self.dropout(decoder3_rough), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_rough = self.instance_normal_de_4_rough(self.deconv4_rough(self.leaky_relu(decoder3_rough)))
		# [batch,1024,h/16,w/16]
		decoder4_rough = torch.cat((decoder4_rough, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_rough = self.instance_normal_de_5_rough(self.deconv5_rough(self.leaky_relu(decoder4_rough)))
		# [batch,512,h/8,w/8]
		decoder5_rough = torch.cat((decoder5_rough, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_rough = self.instance_normal_de_6_rough(self.deconv6_rough(self.leaky_relu(decoder5_rough)))
		# [batch,256,h/4,w/4]
		decoder6_rough = torch.cat((decoder6_rough, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_rough = self.instance_normal_de_7_rough(self.deconv7_rough(self.leaky_relu(decoder6_rough)))
		# [batch,128,h/2,w/2]
		decoder7_rough = torch.cat((decoder7_rough, encoder1), 1)

		# [batch,_out_c,h,w]
		decoder8_rough = self.deconv8_rough(self.leaky_relu(decoder7_rough))

		rough = self.tan(decoder8_rough)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_spec = self.instance_normal_de_1_spec(self.deconv1_spec(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_spec = torch.cat((self.dropout(decoder1_spec), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_spec = self.instance_normal_de_2_spec(self.deconv2_spec(self.leaky_relu(decoder1_spec)))
		# [batch,1024,h/64,w/64]
		decoder2_spec = torch.cat((self.dropout(decoder2_spec), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_spec = self.instance_normal_de_3_spec(self.deconv3_spec(self.leaky_relu(decoder2_spec)))
		# [batch,1024,h/32,w/32]
		decoder3_spec = torch.cat((self.dropout(decoder3_spec), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_spec = self.instance_normal_de_4_spec(self.deconv4_spec(self.leaky_relu(decoder3_spec)))
		# [batch,1024,h/16,w/16]
		decoder4_spec = torch.cat((decoder4_spec, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_spec = self.instance_normal_de_5_spec(self.deconv5_spec(self.leaky_relu(decoder4_spec)))
		# [batch,512,h/8,w/8]
		decoder5_spec = torch.cat((decoder5_spec, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_spec = self.instance_normal_de_6_spec(self.deconv6_spec(self.leaky_relu(decoder5_spec)))
		# [batch,256,h/4,w/4]
		decoder6_spec = torch.cat((decoder6_spec, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_spec = self.instance_normal_de_7_spec(self.deconv7_spec(self.leaky_relu(decoder6_spec)))
		# [batch,128,h/2,w/2]
		decoder7_spec = torch.cat((decoder7_spec, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_spec = self.deconv8_spec(self.leaky_relu(decoder7_spec))

		spec = self.tan(decoder8_spec)

		if self.rough_nc==1:
			rough=rough.repeat(1,3,1,1)

		output=torch.cat((normal,diff,rough,spec),1)

		# print('shape: ',output.shape)

		########################################## Estimate Light #################################################################
		#[B,1,512]
		flat_encoder8=encoder8.view(-1,self.num_flat_features(encoder8))
		#[B,1,256]
		LightPo= self.leaky_relu(self.FC1_Light(flat_encoder8))
		#[B,1,128]
		LightPo= self.leaky_relu(self.FC2_Light(LightPo))
		#[B,1,64]
		LightPo= self.leaky_relu(self.FC3_Light(LightPo))
		#[B,1,32]
		LightPo= self.leaky_relu(self.FC4_Light(LightPo))
		#[B,1,16]
		LightPo= self.leaky_relu(self.FC5_Light(LightPo))
		#[B,1,3]
		LightPo= self.FC6_Light(LightPo)

		return output, LightPo

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class MultiscaleDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
				 use_sigmoid=False, num_D=3, getIntermFeat=False, use_dropout=False):
		super(MultiscaleDiscriminator, self).__init__()
		self.num_D = num_D
		self.n_layers = n_layers
		self.getIntermFeat = getIntermFeat
	 
		for i in range(num_D):
			netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, use_dropout)
			if getIntermFeat:                                
				for j in range(n_layers+2):
					setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
			else:
				setattr(self, 'layer'+str(i), netD.model)

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def singleD_forward(self, model, input):
		if self.getIntermFeat:
			result = [input]
			for i in range(len(model)):
				result.append(model[i](result[-1]))
			return result[1:]
		else:
			return [model(input)]

	def forward(self, input):        
		num_D = self.num_D
		result = []
		input_downsampled = input
		for i in range(num_D):
			if self.getIntermFeat:
				# print('number i: ',i, 'input: ', input_downsampled.shape)

				model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
			else:
				model = getattr(self, 'layer'+str(num_D-1-i))
			result.append(self.singleD_forward(model, input_downsampled))
			if i != (num_D-1):
				input_downsampled = self.downsample(input_downsampled)
		return result
		
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, use_dropout=False):
		super(NLayerDiscriminator, self).__init__()
		self.getIntermFeat = getIntermFeat
		self.n_layers = n_layers

		kw = 4
		padw = int(np.ceil((kw-1.0)/2))
		sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf * 2, 512)
			if use_dropout:
				print('dropout for D')
				sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
				norm_layer(nf), nn.LeakyReLU(0.2, True),nn.Dropout(0.5)
				]]
			else:
				sequence += [[
					nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
					norm_layer(nf), nn.LeakyReLU(0.2, True)
				]]   

		nf_prev = nf
		nf = min(nf * 2, 512)
		if use_dropout:
			print('dropout for D')            
			sequence += [[
			nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
			norm_layer(nf),
			nn.LeakyReLU(0.2, True),nn.Dropout(0.5)
			]]
		else:
			sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
				norm_layer(nf),
				nn.LeakyReLU(0.2, True)
			]]

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		if use_sigmoid:
			sequence += [[nn.Sigmoid()]]

		if getIntermFeat:
			for n in range(len(sequence)):
				setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
		else:
			sequence_stream = []
			for n in range(len(sequence)):
				sequence_stream += sequence[n]
			self.model = nn.Sequential(*sequence_stream)

	def forward(self, input):
		if self.getIntermFeat:
			res = [input]
			for n in range(self.n_layers+2):
				model = getattr(self, 'model'+str(n))
				res.append(model(res[-1]))
			return res[1:]
		else:
			return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)        
		h_relu3 = self.slice3(h_relu2)        
		h_relu4 = self.slice4(h_relu3)        
		h_relu5 = self.slice5(h_relu4)                
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out
