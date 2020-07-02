import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import utils
import loss
from model import Generator, Downsampler, LinearDownsampler, GeneratorWithSparseVector
from skimage.metrics import peak_signal_noise_ratio as psnr 

import argparse
import os
import numpy as np
import cv2
import logging

use_cuda = True
scale = 4
lr_path = './img/LR/Unknown/0217x4.png' 
n_iters = 9501
reg_noise_std = 0.1 #0.03 # standard deviation of added noise after each training set
save_frequency = 50
result_path = './results/DIV2K-Unknown-0217x4-std0.1-weight'

def png2tensor(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def tensor2png(tensor, save_path, filename):
    tensor = tensor.data.squeeze()
    if use_cuda:
        tensor = tensor.cpu()
    pil_img = ToPILImage()(tensor)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, filename)
    pil_img.save(filename)
    print('Saving')

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.data.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def get_noise(input_channel, spatial_size_h=128, spatial_size_w=128, var=1./10):
    spatial_size = (spatial_size_h, spatial_size_w)
    shape = [1, input_channel, spatial_size[0], spatial_size[1]]
    noise_input = torch.zeros(shape)
    noise_input = noise_input.normal_()
    noise_input *= var
    return noise_input

if __name__ == "__main__":
    utils.logger_info('SelfSR-track', log_path='./log/DIV2K-Unknown-LinearDownsample-track.log')
    logger = logging.getLogger('SelfSR-track')

    lr = png2tensor(lr_path)
    h, w = lr.shape[2:]
    net_input = get_noise(3, h, w)
    # noise = net_input
    # net_input_saved = net_input
    # model_G = Generator(scale=scale)
    model_G = GeneratorWithSparseVector(scale=scale)
    # model_D = Downsampler(scale=scale)
    model_D = LinearDownsampler(scale=scale)
    
    if use_cuda:
        net_input = Variable(net_input)
        net_input = net_input.cuda()
        # noise = noise.cuda()
        # net_input_saved = net_input_saved.cuda()
        lr = lr.cuda()
        model_G = model_G.cuda()
        model_D = model_D.cuda()

    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-4)

    l1 = nn.L1Loss()
    ltloss = loss.LTLoss()

    for iter in range(n_iters):
        out_hr = model_G(net_input, lr)
        out_lr = model_D(out_hr)
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        
        bicubic_hr = F.interpolate(lr, scale_factor=scale, mode='bicubic')

        # loss = l1(out_lr, lr) + l1(out_hr, bicubic_hr)
        loss = l1(out_lr, lr) + 0.8*ltloss(out_hr, bicubic_hr) 
        # loss = l1(out_lr, lr)
        loss.backward()
        optimizer_G.step()
        optimizer_D.step()

        print('At step {:05d}, loss is {:.4f}'.format(iter, loss.data.cpu()))

        psnr_LR = psnr(tensor2np(out_lr), tensor2np(lr))
        logger.info("At step {:05d}, out_lr's psnr is {:.4f}".format(iter, psnr_LR))
        
        if iter % save_frequency == 0:
            sr_path = os.path.join(result_path, 'SR/X{}'.format(scale))
            lr_path = os.path.join(result_path, 'LR/X{}'.format(scale))
            tensor2png(out_hr.data, sr_path, 'out_hr_{:04d}.png'.format(iter))
            tensor2png(out_lr.data, lr_path, 'out_lr_{:04d}.png'.format(iter))
            
        # adding noise after each training set
        if reg_noise_std > 0:
            if use_cuda:
                net_input  += (net_input.normal_() * reg_noise_std).cuda()
            else:
                net_input  += (net_input.normal_() * reg_noise_std)

    #clean up any mess we're leaving on the gpu
    if use_cuda:
        torch.cuda.empty_cache()
            




        

    

