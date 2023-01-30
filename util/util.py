from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

EPSILON=1e-7
Threshold=0.01
def InvariantScaling(x,y,max_loop):
    es=1

    for i in range(max_loop):
        # print('before', torch.mean(x.detach()))
        curX = x.detach()*es
        # print('after',torch.mean(curX))
        curX = torch.clamp(curX,0.01,1)
        mean=torch.mean(torch.div(y.detach(),curX))
        es = mean*es

        if torch.abs(mean-1)<0.0001:
            break

        # if es>1.1 or torch.isnan(es):
        #     print("error happen when scaling!!")
        #     print(es)
        #     # break
        #     # es=1
        #     # break
        #     raise("error happen when scaling!!")
        
    es = 1.2 if es>1.2 else es
    es = 0.8 if es<0.8 else es
    return torch.clamp(x*es,0.01,1)


def logTensor(inputimage):
    return  (torch.log(inputimage+0.01) - np.log(0.01)) / (np.log(1.01)-np.log(0.01))


def VGGpreprocess(x):  
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(-1).unsqueeze(-1)
    return (x-mean)/std

def lognp(inputimage):
    return  (np.log(inputimage+0.01) - np.log(0.01)) / (np.log(1.01)-np.log(0.01))

def imshow(img):
    if isinstance(img,(np.ndarray))==False:
        npimg = img.numpy()
    else:
        npimg=img

    # [B,W,H,C]
    if npimg.ndim==4:
        npimg = npimg[0,:,:,:]
    # [W,H,C]
    elif npimg.ndim==3:
        npimg = npimg
    # [B,N,W,H,C]
    elif npimg.ndim==5:
        npimg = npimg[0,0,:,:,:]

    plt.axis("off")
    npimg = npimg*255
    npimg = npimg.clip(0,255)
    #npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    # print('size in imshow: ',npimg.shape)
    plt.imshow(npimg)
    # plt.show()

# [-1,1] -> [0,1]
def tensorNormalize(image_tensor, imtype=np.uint8):
    return ( image_tensor + 1) / 2.0 

def NormMeanStd(image_tensor, Mean, Std):
    return ( image_tensor - Mean) / Std 

def Inverese_NormMeanStd(image_tensor, Mean, Std):
    return image_tensor*Std + Mean 

def tensor2im(image_tensor, imtype=np.uint8, normalize=True, gamma=False, InverseLog=False):
    
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 
        
        if InverseLog:
            temp=image_numpy*(np.log(1.01)-np.log(0.01))+np.log(0.01)
            image_numpy=np.exp(temp)-0.01

        if gamma:
            image_numpy=image_numpy**(1/2.2)* 255.0
        else:
            image_numpy=image_numpy* 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if gamma:
            image_numpy=image_numpy**(1/2.2)* 255.0
        else:
            image_numpy=image_numpy* 255.0  

    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)



def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)
    image_pil.save(image_path, format='JPEG', subsampling=0, quality=100)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

