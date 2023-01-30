import os.path
import os
from data.base_dataset import BaseDataset, get_params, get_transform, normalize,get_transform_real
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from util.util import logTensor,lognp
import random

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.MyTest=opt.MyTest
        self.mode=opt.mode

        if opt.MyTest!='NA':
            self.dir_A = os.path.join(opt.dataroot)  # get the image directory
            self.A_paths = sorted(make_dataset(self.dir_A))
        else:
            ### input A (label maps)
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

            ### input B (real images)
            if opt.isTrain:
                dir_B = '_B' if self.opt.label_nc == 0 else '_img'
                self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
                self.B_paths = sorted(make_dataset(self.dir_B))
 
        self.dataset_size = len(self.A_paths)
      
    def __getitem__(self, index): 

        ### MyTest for diffuse
        if self.MyTest=='Diff':
            ## load images
            A_path = self.A_paths[index]              
            AB = Image.open(A_path).convert('RGB')
            w, h = AB.size 
            ## crop to obtain imageA(input) and imageB(diff)             
            w5 = int(w / 5)
            A = AB.crop((0, 0, w5, h))
            Nor = AB.crop((w5, 0, 2*w5, h))    
            Dif = AB.crop((2*w5, 0, 3*w5, h))    
            Rou = AB.crop((3*w5, 0, 4*w5, h))    
            Spe = AB.crop((4*w5, 0, 5*w5, h))

            ## gamma correction for both A and B
            gamma_A=255.0*(np.array(A)/255.0)**(1/2.2)
            # gamma_B=255.0*(np.array(B)/255.0)**(1/2.2)
            A=Image.fromarray(np.uint8(gamma_A))

            ## transform A and B in the same way
            params = get_params(self.opt, A.size)
            transform = get_transform(self.opt, params)
            A_tensor = transform(A)
            Nor_tensor = transform(Nor)
            Dif_tensor = transform(Dif)
            Rou_tensor = transform(Rou)
            Spe_tensor = transform(Spe)

            B_tensor=torch.cat((Nor_tensor,Dif_tensor,Rou_tensor,Spe_tensor), 0)
        elif self.MyTest=='Normal':

            A_path = self.A_paths[index]              
            AB = Image.open(A_path).convert('RGB') 
            w, h = AB.size 
            ## crop to obtain imageA(input) and imageB(normal)             
            w5 = int(w / 5)
            A = AB.crop((0, 0, w5, h))
            B = AB.crop((w5, 0, 2*w5, h))    
            ## gamma correction for only A
            gamma_A=255.0*(np.array(A)/255.0)**(1/2.2)
            A=Image.fromarray(np.uint8(gamma_A))
            ## transform A and B in the same way

            params = get_params(self.opt, A.size)
            transform = get_transform(self.opt, params)
            A_tensor = transform(A)
            B_tensor = transform(B)
        elif self.MyTest=='Spec':

            A_path = self.A_paths[index]              
            AB = Image.open(A_path).convert('RGB') 
            w, h = AB.size 
            ## crop to obtain imageA(input) and imageB(normal)             
            w5 = int(w / 5)
            A = AB.crop((0, 0, w5, h))
            B = AB.crop((4*w5, 0, 5*w5, h))    
            ## gamma correction for only A
            gamma_A=255.0*(np.array(A)/255.0)**(1/2.2)
            A=Image.fromarray(np.uint8(gamma_A))
            ## transform A and B in the same way

            params = get_params(self.opt, A.size)
            transform = get_transform(self.opt, params)
            A_tensor = transform(A)
            B_tensor = transform(B)
        elif self.MyTest=='Rough':

            A_path = self.A_paths[index]              
            AB = Image.open(A_path).convert('RGB') 
            w, h = AB.size 
            ## crop to obtain imageA(input) and imageB(normal)             
            w5 = int(w / 5)
            A = AB.crop((0, 0, w5, h))
            B = AB.crop((3*w5, 0, 4*w5, h))    
            ## gamma correction for only A
            gamma_A=255.0*(np.array(A)/255.0)**(1/2.2)
            A=Image.fromarray(np.uint8(gamma_A))
            ## transform A and B in the same way

            params = get_params(self.opt, A.size)
            transform = get_transform(self.opt, params)
            A_tensor = transform(A)
            B_tensor = transform(B)
        elif self.MyTest=='NA':
            ### input A (label maps)
            A_path = self.A_paths[index]              
            A = Image.open(A_path)        
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0

            ### input B (real images)
            if self.opt.isTrain or self.opt.use_encoded_image:
                B_path = self.B_paths[index]   
                B = Image.open(B_path).convert('RGB')
                transform_B = get_transform(self.opt, params)      
                B_tensor = transform_B(B)
        # for all_4D and all_5D cases
        else:
            if self.mode=='Syn':
                A_path = self.A_paths[index]    
                # print('syn index', index)          
                AB = Image.open(A_path).convert('RGB') 
                w, h = AB.size 
                ## crop to obtain imageA(input) and imageB(normal)             
                w5 = int(w / 5)
                A = AB.crop((0, 0, w5, h))
                Nor = AB.crop((w5, 0, 2*w5, h))    
                Dif = AB.crop((2*w5, 0, 3*w5, h))    
                Rou = AB.crop((3*w5, 0, 4*w5, h))    
                Spe = AB.crop((4*w5, 0, 5*w5, h))    

                ## gamma correction for only A
                gamma_A=255.0*(np.array(A)/255.0)**(1/2.2)
                ## log scale
                # gamma_A=255.0*logTensor(np.array(A)/255.0)
                A=Image.fromarray(np.uint8(gamma_A))

               ## transform A and B in the same way
                params = get_params(self.opt, A.size)
                transform = get_transform(self.opt, params)
                A_tensor = transform(A)
                Nor_tensor = transform(Nor)
                Dif_tensor = transform(Dif)
                Rou_tensor = transform(Rou)
                Spe_tensor = transform(Spe)

                B_tensor=torch.cat((Nor_tensor,Dif_tensor,Rou_tensor,Spe_tensor), 0)
                # print('norm: ',B_tensor.shape)
            
            ## this is for real images testing
            elif self.mode=='Real':

                A_path = self.A_paths[index]              
                A = Image.open(A_path).convert('RGB') 

                ## transform A and B in the same way
                params = get_params(self.opt, A.size)
                transform = get_transform(self.opt, params,normalize=False)
                A_tensor = transform(A)
                B_tensor = 0

        return {'label': A_tensor, 'image': B_tensor, 'path': A_path}

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'



class AlignedDataset_Real(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root_dir = opt.real_dataroot
        self.dir_list1 = [x for x in os.listdir(opt.real_dataroot) if os.path.isdir(os.path.join(opt.real_dataroot,x))]
        self.dir_number = len(self.dir_list1)
        print('the length of real images data is ', self.dir_number)

    def __getitem__(self, index): 

        dir_path = os.path.join(self.root_dir,self.dir_list1[index])
        image_list = [x for x in os.listdir(dir_path)]
        image_number = len(image_list)

        image_index = random.sample(range(image_number),2)

        # load real image pairs
        A_path = os.path.join(dir_path,image_list[image_index[0]])
        B_path = os.path.join(dir_path,image_list[image_index[1]])
        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')
        w,h = image_A.size
        params = get_params(self.opt, image_A.size)
        transform = get_transform_real(self.opt, params, normalize=False)
        A_tensor = transform(image_A)
        B_tensor = transform(image_B)

        return {'label': A_tensor, 'image': B_tensor, 'path': A_path}

    def __len__(self):
        return self.dir_number

    def name(self):
        return 'AlignedDataset_Real'



class AlignedDataset_Process(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root_dir = opt.real_dataroot
        self.dir_list1 = [x for x in os.listdir(opt.real_dataroot) if os.path.isdir(os.path.join(opt.real_dataroot,x))]
        self.dir_number = len(self.dir_list1)

    def __getitem__(self, index): 

        dir_path = os.path.join(self.root_dir,self.dir_list1[index])
        image_list = [x for x in os.listdir(dir_path)]
        image_number = len(image_list)
        # print('index: ', index, 'list number: ', image_number)

        image_index = random.sample(range(image_number),2)

        # load real image pairs
        A_path = os.path.join(dir_path,image_list[image_index[0]])
        B_path = os.path.join(dir_path,image_list[image_index[1]])

        image_A = Image.open(A_path).convert('RGB')
        image_B = Image.open(B_path).convert('RGB')

        w,h = image_A.size

        # gamma correction for both A and B
        # gamma_A = 255.0*(np.array(image_A)/255.0)**(2.2)
        # gamma_B = 255.0*(np.array(image_B)/255.0)**(2.2)
        # A = Image.fromarray(np.uint8(gamma_A))       
        # B = Image.fromarray(np.uint8(gamma_B))       
        
        params = get_params(self.opt, image_A.size)
        transform = get_transform_real(self.opt, params)
        A_tensor = transform(image_A)
        B_tensor = transform(image_B)

        input_dict = {'label': A_tensor, 'inst': 0, 'image': B_tensor, 
                      'feat': 0, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dir_number

    def name(self):
        return 'AlignedDataset_Process'