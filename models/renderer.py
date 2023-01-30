import numpy as np
import torch
import random
import sys

import torchvision.transforms.functional as TF

from util import *
import matplotlib.pyplot as plt
import torch.distributions as tdist
from numpy import nan

PI=np.pi
EPSILON=1e-7


def PositionMap(width,height,channel):

	Position_map_cpu = torch.zeros((width,height,channel))
	for w in range(width):
		for h in range(height):
			Position_map_cpu[h][w][0] = 2*w/(width-1) - 1
			#Position_map[h][w][0] = 2*(width-w-1)/(width-1) - 1
			Position_map_cpu[h][w][1] = 2*(height-h-1)/(height-1) - 1
			#Position_map[h][w][1] = 2*h/(height-1) - 1
			Position_map_cpu[h][w][2] = 0

	return Position_map_cpu
# mydevice = torch.device("cuda")

# seed=1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) 

# random.seed(seed)
# np.random.seed(seed)

############################################## Creating Light & Camera Positions ################################################
#### randomly sample one direction
def Cosine_Distribution(mydevice):

	# u_1=torch.tensor([0.1],device=mydevice)
	u_1= torch.rand(1,device=mydevice)*0.95+0.001
	# print('u_1:', u_1)

	# u_2=torch.tensor([0.5],device=mydevice)
	u_2= torch.rand(1,device=mydevice)
	# print('u_2:', u_2)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.tensor([x,y,z],device=mydevice)

	# print(temp_out.shape)

	return temp_out
   

##### the input of normalize_vec function must be an image with 3 channels
def normalize_vec(input_tensor):
	
	## speed test manually vs norm()
	if input_tensor.dim() == 3:  
		# NormalizedNorm_len=input_tensor[:,:,0]*input_tensor[:,:,0]+input_tensor[:,:,1]*input_tensor[:,:,1]+input_tensor[:,:,2]*input_tensor[:,:,2]
		# NormalizedNorm_len=torch.sqrt(NormalizedNorm_len)
		NormalizedNorm_len=torch.norm(input_tensor,2,2)
		# print('shape:',NormalizedNorm_len.shape)
		NormalizedNorm = input_tensor/(NormalizedNorm_len[:,:,np.newaxis]+EPSILON)
		return NormalizedNorm
	elif input_tensor.dim() == 4:
		NormalizedNorm_len=input_tensor[:,:,:,0]*input_tensor[:,:,:,0]+input_tensor[:,:,:,1]*input_tensor[:,:,:,1]+input_tensor[:,:,:,2]*input_tensor[:,:,:,2]
		NormalizedNorm_len=torch.sqrt(NormalizedNorm_len)
		# NormalizedNorm_len=torch.norm(input_tensor,2,3)
		# print('shape:',NormalizedNorm_len.shape)
		NormalizedNorm = input_tensor/(NormalizedNorm_len[:,:,:,np.newaxis]+EPSILON)
		return NormalizedNorm
	else:
		print('incorrectly input')
		return


#### randomly sample Number direction (for the rendering in backpropgation during training )
def Cosine_Distribution_Number(Number, r_max, mydevice):

	mydevice=torch.device('cuda')

	u_1= torch.rand((Number,1),device=mydevice)*r_max+0.001 	# rmax: 0.95 (default)
	# print('u_1:', u_1)

	u_2= torch.rand((Number,1),device=mydevice)
	# print('u_2:', u_2)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.cat([x,y,z],1)

	# print('tttt: ',temp_out.shape)

	return temp_out

def Create_LightCamera_Position(Near_Number, Dist_Number, r_max, mydevice):

	Mirror_tensor=torch.tensor([-1.0,-1.0,1.0],device=mydevice)
	Mirror_tensor=Mirror_tensor.repeat(Near_Number,1)
	################## random light and camera position ########################
	## Case 1: distant light view directions:
	if Dist_Number>0:
		Dist_light = Cosine_Distribution_Number(Dist_Number,r_max,mydevice)
		Dist_view = Cosine_Distribution_Number(Dist_Number,r_max,mydevice)

	## Case 2: cosine distribution of view and light 
	rand_light = Cosine_Distribution_Number(Near_Number,r_max,mydevice)
	rand_view = rand_light*Mirror_tensor

	## randomly select one distance normal distribution
	# Distance=torch.tensor([0.5,0.5],device=mydevice)
	m=tdist.Normal(torch.tensor([0.5]),torch.tensor([0.75]))
	Distance=m.sample((Near_Number,2)).to(mydevice)

	# Origin=torch.tensor([0.0,0.0],device=mydevice)
	Origin = torch.rand((Near_Number,2),device=mydevice)*2-1
	Origin = torch.cat([Origin,torch.zeros((Near_Number,1),device=mydevice)],1)

	rand1_Light_po=Origin+rand_light*torch.exp(Distance[:,0])
	rand2_Camera_po=Origin+rand_view*torch.exp(Distance[:,1])

	############ setting light and camera position #########################
	# random light and camera
	light_po_gpu=rand1_Light_po
	camera_po_gpu=rand2_Camera_po

	return light_po_gpu,camera_po_gpu,Dist_light,Dist_view


def Create_NumberPointLightPosition(Near_Number,r_max, mydevice):

	mydevice=torch.device('cuda')

	rand_light = Cosine_Distribution_Number(Near_Number, r_max, mydevice)

	# Origin ([-1,1],[-1,1],0)
	# Origin=torch.tensor([0.0,0.0],device=mydevice)
	# Origin_xy = torch.rand((Near_Number,2),device=mydevice)*2-1
	Origin_xy = torch.rand((Near_Number,2),device=mydevice)*0
	Origin = torch.cat([Origin_xy,torch.zeros((Near_Number,1),device=mydevice)],1)

	m=tdist.Normal(torch.tensor([1.0]),torch.tensor([0.75]))
	Distance=m.sample((Near_Number,2)).to(mydevice)
	Light_po=Origin+rand_light*torch.exp(Distance[:,0])

	return Light_po


############################################# Rendering Function ###################################################################
# Single Render: each scene rendered under single light (camera) positin
# Batch Render: each scene rendered under multiple light (camera) position at the same time

## render algorithm
def Rendering_Algorithm(diff,spec,rough,NdotH,NdotL,VdotN,VdotH):

	NdotH=torch.clamp(NdotH,0,1)
	NdotL=torch.clamp(NdotL,0,1)
	VdotN=torch.clamp(VdotN,0,1)
	VdotH=torch.clamp(VdotH,0,1)

	########################### rendering alogorithm ###########################################################################
	diff=diff*(1-spec)/PI

	## only one channel for roughness
	rough2 = rough*rough
	NdotH2 = NdotH*NdotH
	
	### psudocode: GGX distribution with PI removed
	#### D #####
	deno_D = torch.max((rough2*rough2 - 1)*NdotH2+1,torch.tensor([0.00001]).cuda())
	D = (rough2/deno_D)*(rough2/deno_D)
	D = D/PI

	#### G #####
	G_1 = 1/(NdotL*(1-rough2/2)+rough2/2+EPSILON)
	G_2 = 1/(VdotN*(1-rough2/2)+rough2/2+EPSILON)
	G = G_1*G_2

	#### F #####
	F = spec+(1-spec)*2**((-5.55473*VdotH - 6.98316)*VdotH)

	specular = G*F*D/(4+EPSILON)

	# [B,N,W,H,C]
	FinalColor = PI*(diff+specular)*NdotL ## this is for testing??
	# FinalColor = PI*(diff+specular)*NdotL/(L_dot_verticalnorm+EPSILON) ## training equation (L*Norm used to compensate the graze angle)

	return FinalColor

## paper model
def tf_Render(diffuse,specular,roughness,NdotH,NdotL,NdotV,VdotH):

	print('tf render')

	def tf_render_diffuse_Substance(diffuse, specular):
		return diffuse * (1.0 - specular) / PI

	def tf_render_D_GGX_Substance(roughness, NdotH):
		alpha = torch.pow(roughness,2)
		underD = 1/torch.max((torch.pow(NdotH,2) * (torch.pow(alpha,2) - 1.0) + 1.0),torch.tensor([0.00001]).cuda())
		return (torch.pow(alpha * underD,2)/PI)
		

	def tf_render_F_GGX_Substance(specular, VdotH):
		sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
		return specular + (1.0 - specular) * sphg
		
	def tf_render_G_GGX_Substance(roughness, NdotL, NdotV):
		
		def G1_Substance(NdotW, k):
			return 1.0/torch.max((NdotW * (1.0 - k) + k), torch.tensor([0.00001]).cuda())

		return G1_Substance(NdotL, torch.pow(roughness,2)/2) * G1_Substance(NdotV, torch.pow(roughness,2)/2)
		

	diffuse_rendered = tf_render_diffuse_Substance(diffuse, specular)
	D_rendered = tf_render_D_GGX_Substance(roughness, NdotH)
	G_rendered = tf_render_G_GGX_Substance(roughness, NdotL, NdotV)
	F_rendered = tf_render_F_GGX_Substance(specular, VdotH)
	
	
	specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
	result = specular_rendered
	
	result = result + diffuse_rendered
	
	lampIntensity = 1.0
	
	lampFactor = lampIntensity * PI#tf_lampAttenuation(lampDistance) * lampIntensity * math.pi
	
	result = result * lampFactor

	result = result *  NdotL# / tf.expand_dims(tf.maximum(wiNorm[:,:,:,2], 0.001), axis=-1) # This division is to compensate for the cosinus distribution of the intensity in the rendering
			
	return result#[result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]

## Rendering function is computed on GPU
## render under single light and camera Position
## input: feature: [B,W,H,C]; light/camera [3]
## output: [B,W,H,C]
def SingleRender(diff_image, spec_image, normal_image, roughness_image, LightPosition, Position_map,mydevice, low_cam):

	# mydevice=torch.device('cuda')
	if low_cam:
		CameraPosition=torch.tensor([0.,0.,1.00],device=mydevice)
	else:
		CameraPosition=torch.tensor([0.,0.,2.14],device=mydevice)

	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here
		
	# gpu_tracker.track()
	# vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)

	############################ Light view and H Vector #####################################
	## light vector
	L_vec = LightPosition - Position_map
	##distant lighting
	# L_vec = Cosine_Distribution(mydevice).repeat(width,height,1)
	L_vec_norm = normalize_vec(L_vec)

	## view vector
	V_vec = CameraPosition - Position_map
	##distant lighting
	# V_vec = Cosine_Distribution(mydevice).repeat(width,height,1)
	V_vec_norm = normalize_vec(V_vec)
	## Half vector of view and light direction
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	L_vec_norm=L_vec_norm.repeat(batch,1,1,1)
	V_vec_norm=V_vec_norm.repeat(batch,1,1,1)
	H_vec_norm=H_vec_norm.repeat(batch,1,1,1)
	# vertical_norm=vertical_norm.repeat(batch,width,height,1)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	Normal_vec = normalize_vec(normal)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	# L_dot_verticalnorm=L_vec_norm*vertical_norm

	## manually computation faster
	NdotL=(NdotL[:,:,:,0]+NdotL[:,:,:,1]+NdotL[:,:,:,2]).reshape((batch,width,height,1))
	NdotH=(NdotH[:,:,:,0]+NdotH[:,:,:,1]+NdotH[:,:,:,2]).reshape((batch,width,height,1))
	VdotH=(VdotH[:,:,:,0]+VdotH[:,:,:,1]+VdotH[:,:,:,2]).reshape((batch,width,height,1))
	VdotN=(VdotN[:,:,:,0]+VdotN[:,:,:,1]+VdotN[:,:,:,2]).reshape((batch,width,height,1))
	# L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,0:1]+L_dot_verticalnorm[:,:,:,1:2]+L_dot_verticalnorm[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	# NdotL = torch.ones((batch,width,height,1),device=mydevice)
	# NdotH = torch.ones((batch,width,height,1),device=mydevice)
	# VdotH = torch.ones((batch,width,height,1),device=mydevice)
	# VdotN = torch.ones((batch,width,height,1),device=mydevice)
	#####################################################################################

	rough = roughness_image[:,:,:,0:1]

	NdotH=torch.clamp(NdotH,0,1)
	NdotL=torch.clamp(NdotL,0,1)
	VdotN=torch.clamp(VdotN,0,1)
	VdotH=torch.clamp(VdotH,0,1)

	FinalColor=Rendering_Algorithm(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	
	return FinalColor
	

def SingleRender_camera(diff_image, spec_image, normal_image, roughness_image, LightPosition,CameraPosition, Position_map,mydevice):

	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here
		
	# gpu_tracker.track()
	# vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)

	############################ Light view and H Vector #####################################
	## light vector
	L_vec = LightPosition - Position_map
	##distant lighting
	# L_vec = Cosine_Distribution(mydevice).repeat(width,height,1)
	L_vec_norm = normalize_vec(L_vec)

	## view vector
	V_vec = CameraPosition - Position_map
	##distant lighting
	# V_vec = Cosine_Distribution(mydevice).repeat(width,height,1)
	V_vec_norm = normalize_vec(V_vec)
	## Half vector of view and light direction
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	L_vec_norm=L_vec_norm.repeat(batch,1,1,1)
	V_vec_norm=V_vec_norm.repeat(batch,1,1,1)
	H_vec_norm=H_vec_norm.repeat(batch,1,1,1)
	# vertical_norm=vertical_norm.repeat(batch,width,height,1)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	Normal_vec = normalize_vec(normal)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	# L_dot_verticalnorm=L_vec_norm*vertical_norm

	## manually computation faster
	NdotL=(NdotL[:,:,:,0]+NdotL[:,:,:,1]+NdotL[:,:,:,2]).reshape((batch,width,height,1))
	NdotH=(NdotH[:,:,:,0]+NdotH[:,:,:,1]+NdotH[:,:,:,2]).reshape((batch,width,height,1))
	VdotH=(VdotH[:,:,:,0]+VdotH[:,:,:,1]+VdotH[:,:,:,2]).reshape((batch,width,height,1))
	VdotN=(VdotN[:,:,:,0]+VdotN[:,:,:,1]+VdotN[:,:,:,2]).reshape((batch,width,height,1))
	# L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,0:1]+L_dot_verticalnorm[:,:,:,1:2]+L_dot_verticalnorm[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	# NdotL = torch.ones((batch,width,height,1),device=mydevice)
	# NdotH = torch.ones((batch,width,height,1),device=mydevice)
	# VdotH = torch.ones((batch,width,height,1),device=mydevice)
	# VdotN = torch.ones((batch,width,height,1),device=mydevice)
	#####################################################################################

	rough = roughness_image[:,:,:,0:1]

	NdotH=torch.clamp(NdotH,0,1)
	NdotL=torch.clamp(NdotL,0,1)
	VdotN=torch.clamp(VdotN,0,1)
	VdotH=torch.clamp(VdotH,0,1)

	FinalColor=Rendering_Algorithm(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	
	return FinalColor



## render under multiple light positions (Just Point Light) and fixed camera position (0,0,1)
## Each scene under single different light position; in each batch there are B scenes and N light positions
## input:image:[B,W,H,C]; LightPosition: [N,3]; Position_map: [W,H,3]; LightIntensity: constant
## output: [B/N,W,H,C]. (B==N)
## !!! B (batch size) must equal N (Number of point light position) !!!
def SingleRender_NumberPointLight_FixedCamera(diff_image, spec_image, normal_image, roughness_image, LightPosition, Position_map, mydevice, Near_Number, CoCamLi):
	
	# mydevice=torch.device('cuda')
	if diff_image.dim() !=4 or spec_image.dim() !=4 or normal_image.dim() !=4 or roughness_image.dim() !=4:
		print('dimention error, your input dimention is: ', diff_image.dim())
		return 

	vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)
	
	# if low_cam:
	# 	CameraPosition=torch.tensor([0.,0.,1.],device=mydevice)
	# else:
	# 	CameraPosition=torch.tensor([0.,0.,2.14],device=mydevice)

	if not CoCamLi:
		# CameraPosition=torch.tensor([0.,0.,2.14],device=mydevice)
		CameraPosition=torch.tensor([0.,0.,2.14],device=mydevice).unsqueeze(0).repeat(Near_Number,1)
	else:
		CameraPosition=LightPosition.clone()


	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here

	# if batch != Near_Number:
	# 	print(batch,' batch size not match light postion size ', Near_Number)
	# 	sys.exit(1)

	Position_map=Position_map.repeat(Near_Number,1,1,1)

	############################ Light view and H Vector #####################################
	## light vector [N,W,H,C]
	L_vec=torch.zeros((Near_Number,width,height,channel),device=mydevice)
	for i in range(Near_Number):
		L_vec[i,:,:,:]=LightPosition[i,:]-Position_map[i,:,:,:]
	L_vec_norm = normalize_vec(L_vec)
	# print(L_vec_norm)

	## view vector [N,W,H,C]
	V_vec=torch.zeros((Near_Number,width,height,channel),device=mydevice)
	for i in range(Near_Number):
		V_vec[i,:,:,:]=CameraPosition[i,:]-Position_map[i,:,:,:]
	V_vec_norm = normalize_vec(V_vec)

	## Half vector of view and light direction [N,W,H,C]
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	## [N,W,H,C] -> [B/N,W,H,C]
	# L_vec_norm=L_vec_norm.repeat(batch,1,1,1,1)
	# V_vec_norm=V_vec_norm.repeat(batch,1,1,1,1)
	# H_vec_norm=H_vec_norm.repeat(batch,1,1,1,1)
	vertical_norm=vertical_norm.repeat(batch,width,height,1)

	# print('temp shape: ', vertical_norm.shape)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	# [B,W,H,C] -> [B,N,W,H,C]
	# Normal_vec = normalize_vec(normal).repeat(Near_Number,1,1,1,1).permute(1,0,2,3,4)
	Normal_vec = normalize_vec(normal)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	# [B/N,W,H,C]
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	L_dot_verticalnorm=L_vec_norm*vertical_norm
	
	## manually computation faster
	# [B/N,W,H,C] -> [B/N,W,H,1]
	NdotL=(NdotL[:,:,:,0:1]+NdotL[:,:,:,1:2]+NdotL[:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	NdotH=(NdotH[:,:,:,0:1]+NdotH[:,:,:,1:2]+NdotH[:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	VdotH=(VdotH[:,:,:,0:1]+VdotH[:,:,:,1:2]+VdotH[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	VdotN=(VdotN[:,:,:,0:1]+VdotN[:,:,:,1:2]+VdotN[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,0:1]+L_dot_verticalnorm[:,:,:,1:2]+L_dot_verticalnorm[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	rough = roughness_image[:,:,:,0:1]#.repeat(Near_Number,1,1,1,1).permute(1,0,2,3,4)

	NdotH=torch.clamp(NdotH,0,1)
	NdotL=torch.clamp(NdotL,0,1)
	VdotN=torch.clamp(VdotN,0,1)
	VdotH=torch.clamp(VdotH,0,1)

	FinalColor=Rendering_Algorithm(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	# FinalColor=tf_Render(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	
	return FinalColor


## !!! B (batch size) must equal N (Number of point light position) !!!
def SingleRender_NumberPointLightCamera(diff_image, spec_image, normal_image, roughness_image, LightPosition,CameraPosition, Position_map, mydevice, Near_Number):
	
	# mydevice=torch.device('cuda')

	if diff_image.dim() !=4 or spec_image.dim() !=4 or normal_image.dim() !=4 or roughness_image.dim() !=4:
		print('dimention error, your input dimention is: ', diff_image.dim())
		return 

	# vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)

	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here

	# if batch != Near_Number:
	# 	print(batch,' batch size not match light postion size ', Near_Number)
	# 	sys.exit(1)

	Position_map=Position_map.repeat(Near_Number,1,1,1)

	############################ Light view and H Vector #####################################
	## light vector [N,W,H,C]
	L_vec=torch.zeros((Near_Number,width,height,channel),device=mydevice)
	for i in range(Near_Number):
		L_vec[i,:,:,:]=LightPosition[i,:]-Position_map[i,:,:,:]
	L_vec_norm = normalize_vec(L_vec)
	# print(L_vec_norm)

	## view vector [N,W,H,C]
	V_vec=torch.zeros((Near_Number,width,height,channel),device=mydevice)
	for i in range(Near_Number):
		V_vec[i,:,:,:]=CameraPosition[i,:]-Position_map[i,:,:,:]
	V_vec_norm = normalize_vec(V_vec)

	## Half vector of view and light direction [N,W,H,C]
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	## [N,W,H,C] -> [B/N,W,H,C]
	# L_vec_norm=L_vec_norm.repeat(batch,1,1,1,1)
	# V_vec_norm=V_vec_norm.repeat(batch,1,1,1,1)
	# H_vec_norm=H_vec_norm.repeat(batch,1,1,1,1)
	# vertical_norm=vertical_norm.repeat(batch,width,height,1)

	# print('temp shape: ', vertical_norm.shape)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	# [B,W,H,C] -> [B,N,W,H,C]
	# Normal_vec = normalize_vec(normal).repeat(Near_Number,1,1,1,1).permute(1,0,2,3,4)
	Normal_vec = normalize_vec(normal)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	# [B/N,W,H,C]
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	# L_dot_verticalnorm=L_vec_norm*vertical_norm
	
	## manually computation faster
	# [B/N,W,H,C] -> [B/N,W,H,1]
	NdotL=(NdotL[:,:,:,0:1]+NdotL[:,:,:,1:2]+NdotL[:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	NdotH=(NdotH[:,:,:,0:1]+NdotH[:,:,:,1:2]+NdotH[:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	VdotH=(VdotH[:,:,:,0:1]+VdotH[:,:,:,1:2]+VdotH[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	VdotN=(VdotN[:,:,:,0:1]+VdotN[:,:,:,1:2]+VdotN[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	# L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,0:1]+L_dot_verticalnorm[:,:,:,1:2]+L_dot_verticalnorm[:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	rough = roughness_image[:,:,:,0:1]#.repeat(Near_Number,1,1,1,1).permute(1,0,2,3,4)

	NdotH=torch.clamp(NdotH,0,1)
	NdotL=torch.clamp(NdotL,0,1)
	VdotN=torch.clamp(VdotN,0,1)
	VdotH=torch.clamp(VdotH,0,1)

	FinalColor=Rendering_Algorithm(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	# FinalColor=tf_Render(diff_image,spec_image,rough,NdotH,NdotL,VdotN,VdotH)
	
	return FinalColor



# render under multiple light camera positions (Point light & distant light)
# input: image:[B,W,H,C]; Position: [N,3]; Position_map: [W,H,3]; LightIntensity: constant
# output: [B,N,W,H,C]
def Batchrender(diff_image, spec_image, normal_image, roughness_image, LightPosition, CameraPosition,Position_map, mydevice, Near_Number,Dist_Number,Dist_light_dir,Dist_view_dir):

	if diff_image.dim() !=4 or spec_image.dim() !=4 or normal_image.dim() !=4 or roughness_image.dim() !=4:
		print('dimention error, your input dimention is: ', diff_image.dim())
		return 

	vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)


	Total_Number=Near_Number+Dist_Number

	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here
	Position_map=Position_map.repeat(Near_Number,1,1,1)

	############################ Light view and H Vector #####################################
	## light vector [N,W,H,C]
	L_vec=torch.zeros((Total_Number,width,height,channel),device=mydevice)
	for i in range(Total_Number):
		if i < Near_Number:
			L_vec[i,:,:,:]=LightPosition[i,:]-Position_map[i,:,:,:]
		elif i>= Near_Number:
			L_vec[i,:,:,:]=Dist_light_dir[i-Near_Number,:].repeat(width,height,1)
	L_vec_norm = normalize_vec(L_vec)
	# print(L_vec_norm)

	## view vector [N,W,H,C]
	V_vec=torch.zeros((Total_Number,width,height,channel),device=mydevice)
	for i in range(Total_Number):
		if i < Near_Number:		
			V_vec[i,:,:,:]=CameraPosition[i,:]-Position_map[i,:,:,:]
		elif i>= Near_Number:
			V_vec[i,:,:,:]=Dist_view_dir[i-Near_Number,:].repeat(width,height,1)
	V_vec_norm = normalize_vec(V_vec)

	## Half vector of view and light direction [N,W,H,C]
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	## [N,W,H,C] -> [B,N,W,H,C]
	L_vec_norm=L_vec_norm.repeat(batch,1,1,1,1)
	V_vec_norm=V_vec_norm.repeat(batch,1,1,1,1)
	H_vec_norm=H_vec_norm.repeat(batch,1,1,1,1)
	vertical_norm=vertical_norm.repeat(batch,Total_Number,width,height,1)

	# print('temp shape: ', vertical_norm.shape)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	# [B,W,H,C] -> [B,N,W,H,C]
	Normal_vec = normalize_vec(normal).repeat(Total_Number,1,1,1,1).permute(1,0,2,3,4)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	# [B,N,W,H,C]
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	L_dot_verticalnorm=L_vec_norm*vertical_norm
	
	## manually computation faster
	# [B,N,W,H,C] -> [B,N,W,H,1]
	NdotL=(NdotL[:,:,:,:,0:1]+NdotL[:,:,:,:,1:2]+NdotL[:,:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	NdotH=(NdotH[:,:,:,:,0:1]+NdotH[:,:,:,:,1:2]+NdotH[:,:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	VdotH=(VdotH[:,:,:,:,0:1]+VdotH[:,:,:,:,1:2]+VdotH[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	VdotN=(VdotN[:,:,:,:,0:1]+VdotN[:,:,:,:,1:2]+VdotN[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,:,0:1]+L_dot_verticalnorm[:,:,:,:,1:2]+L_dot_verticalnorm[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	rough = roughness_image[:,:,:,0:1].repeat(Total_Number,1,1,1,1).permute(1,0,2,3,4)
	s = spec_image.repeat(Total_Number,1,1,1,1).permute(1,0,2,3,4)
	diff = diff_image.repeat(Total_Number,1,1,1,1).permute(1,0,2,3,4)

	# FinalColor1,spec=Rendering_Algorithm(diff,s,rough,NdotH,NdotL,VdotN,VdotH,L_dot_verticalnorm)
	FinalColor1=tf_Render(diff,s,rough,NdotH,NdotL,VdotN,VdotH)
	
	return FinalColor1#,FinalColor2


# render under multple light position (Only Point light) and fixed camera position (0,0,1)
# each scene under multiple light positions
# output [B,N,W,H,C]
def Batchrender_NumberPointLight_FixedCamera(diff_image, spec_image, normal_image, roughness_image, LightPosition,Position_map, mydevice, Number):

	mydevice=torch.device('cuda')

	CameraPosition=torch.tensor([0.,0.,1.],device=mydevice)

	if diff_image.dim() !=4 or spec_image.dim() !=4 or normal_image.dim() !=4 or roughness_image.dim() !=4:
		print('dimention error, your input dimention is: ', diff_image.dim())
		return 

	vertical_norm=torch.tensor([0.,0.,1.],device=mydevice)

	[batch,width,height,channel] = diff_image.shape # the channel is default as 3 here
	Position_map=Position_map.repeat(Number,1,1,1)

	# print(diff_image.shape)
	# print(LightPosition.shape)
	############################ Light view and H Vector #####################################
	## light vector [N,W,H,C]
	L_vec=torch.zeros((Number,width,height,channel),device=mydevice)
	# print(L_vec.shape)

	for i in range(Number):
		L_vec[i,:,:,:]=LightPosition[i,:]-Position_map[i,:,:,:]
	L_vec_norm = normalize_vec(L_vec)
	# print(L_vec_norm)

	## view vector [N,W,H,C]
	V_vec=torch.zeros((Number,width,height,channel),device=mydevice)
	for i in range(Number):
		V_vec[i,:,:,:]=CameraPosition-Position_map[i,:,:,:]
	V_vec_norm = normalize_vec(V_vec)

	## Half vector of view and light direction [N,W,H,C]
	H_vec_norm = normalize_vec((L_vec_norm + V_vec_norm)/2)

	## [N,W,H,C] -> [B,N,W,H,C]
	L_vec_norm=L_vec_norm.repeat(batch,1,1,1,1)
	V_vec_norm=V_vec_norm.repeat(batch,1,1,1,1)
	H_vec_norm=H_vec_norm.repeat(batch,1,1,1,1)
	vertical_norm=vertical_norm.repeat(batch,Number,width,height,1)

	# print('temp shape: ', vertical_norm.shape)

	####################### compute the normal map based on normal_image ################
	### attention !!! remember to *2 -1 first then normalize 
	# [0, 1] => [-1, 1]
	normal = normal_image*2-1
	# [B,W,H,C] -> [B,N,W,H,C]
	Normal_vec = normalize_vec(normal).repeat(Number,1,1,1,1).permute(1,0,2,3,4)
	# print(L_vec_norm.shape)

	## sum() is slower than manually computation
	# [B,N,W,H,C]
	NdotL = (Normal_vec*L_vec_norm)#.sum(3).reshape((batch,width,height,1))
	NdotH = (Normal_vec*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotH = (V_vec_norm*H_vec_norm)#.sum(3).reshape((batch,width,height,1))
	VdotN = (V_vec_norm*Normal_vec)#.sum(3).reshape((batch,width,height,1))
	L_dot_verticalnorm=L_vec_norm*vertical_norm
	
	## manually computation faster
	# [B,N,W,H,C] -> [B,N,W,H,1]
	NdotL=(NdotL[:,:,:,:,0:1]+NdotL[:,:,:,:,1:2]+NdotL[:,:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	NdotH=(NdotH[:,:,:,:,0:1]+NdotH[:,:,:,:,1:2]+NdotH[:,:,:,:,2:3])#.reshape((batch,Total_Number,width,height,1))
	VdotH=(VdotH[:,:,:,:,0:1]+VdotH[:,:,:,:,1:2]+VdotH[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	VdotN=(VdotN[:,:,:,:,0:1]+VdotN[:,:,:,:,1:2]+VdotN[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))
	L_dot_verticalnorm=(L_dot_verticalnorm[:,:,:,:,0:1]+L_dot_verticalnorm[:,:,:,:,1:2]+L_dot_verticalnorm[:,:,:,:,2:3])#.reshape((batch,width,Total_Number,height,1))

	rough = roughness_image[:,:,:,0:1].repeat(Number,1,1,1,1).permute(1,0,2,3,4)
	s = spec_image.repeat(Number,1,1,1,1).permute(1,0,2,3,4)
	diff = diff_image.repeat(Number,1,1,1,1).permute(1,0,2,3,4)

	# [B,N,W,H,C]
	FinalColor=Rendering_Algorithm(diff,s,rough,NdotH,NdotL,VdotN,VdotH,L_dot_verticalnorm)

	FinalColor=FinalColor.permute(0,2,3,1,4).contiguous().view(batch,width,height,-1)

	return FinalColor


