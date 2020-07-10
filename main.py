import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import utils
import loss
from model.arc import Generator, Downsampler
from skimage.metrics import peak_signal_noise_ratio as psnr 
from skimage.metrics import structural_similarity as ssim

import argparse
import os
import numpy as np
import cv2
import logging

use_cuda = True
scale = 4
data_path = '/root/proj/SelfSR/dataset/'
dataset = 'Set14' 
n_iters = 10001
reg_noise_std = 0.5 #0.03 # standard deviation of added noise after each training set
save_frequency = 100
result_path = './results/RCAN-SSL-Set14-std0.5'


if __name__ == "__main__":
    utils.logger_info('SelfSR-track', log_path='./log/Set14-RCAN-SSL-std0.5-track.log')
    logger = logging.getLogger('SelfSR-track')

    lr_path = os.path.join(data_path, dataset, 'LR_bicubic', 'X{}'.format(scale))
    hr_path = os.path.join(data_path, dataset, 'HR')
    
    sr_path = result_path
    if not os.path.exists(sr_path):
        os.makedirs(sr_path)

    model_G = RCAN(scale=scale)
    model_D = LinearDownsampler(scale=scale)
    
    if use_cuda:
        model_G = model_G.cuda()
        model_D = model_D.cuda()

    optimizer = torch.optim.Adam([{'params':model_G.parameters()}, {'params':model_D.parameters()}], lr=2.5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5000, 7000, 9000], gamma=0.5)  # learning rates

    l1 = nn.L1Loss()
    # ltloss = loss.LTLoss()

    filelist = utils.get_list(lr_path)

    idx = 0
    for img in filelist:
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->2d}--> {:>10s}'.format(idx, img_name+ext))
        
        lr = utils.png2tensor(img)
        h, w = lr.shape[2:]
        net_input = utils.get_noise(3, h, w)
        hr = utils.png2tensor(hr_path + '/' + img_name.split('x{}'.format(scale))[0] + ext, scale=scale, crop=True)

        if use_cuda:
            net_input = Variable(net_input)
            net_input = net_input.cuda()
            lr = lr.cuda()
            hr = hr.cuda()

        for iter in range(n_iters):
            out_hr = model_G(net_input)
            out_lr = model_D(out_hr)

            optimizer.zero_grad()
            
            bicubic_hr = F.interpolate(lr, scale_factor=scale, mode='bicubic')

            loss = l1(out_lr, lr) + l1(out_hr, bicubic_hr)
            loss.backward()

            optimizer.step()
            # change the learning rate
            scheduler.step()


            print('At step {:05d}, loss is {:.4f}'.format(iter, loss.data.cpu()))

            psnr_SR = psnr(utils.tensor2np(out_hr), utils.tensor2np(hr))
            ssim_SR = ssim(utils.tensor2np(out_hr), utils.tensor2np(hr), multichannel=True)

            psnr_LR = psnr(utils.tensor2np(out_lr), utils.tensor2np(lr))
            ssim_LR = ssim(utils.tensor2np(out_lr), utils.tensor2np(lr), multichannel=True)
            
            logger.info("At step {:05d}, SR_psnr is {:.4f}, SR_ssim is {:.4f} || LR_psnr is {:.4f}, LR_ssim is {:.4f}".format(iter, psnr_SR, ssim_SR, psnr_LR, ssim_LR))
            
            if iter % save_frequency == 0:
                sr_path = os.path.join(result_path, img_name.split('x{}'.format(scale))[0], 'SR/X{}'.format(scale))
                lr_path = os.path.join(result_path, img_name.split('x{}'.format(scale))[0], 'LR/X{}'.format(scale))
                utils.tensor2png(out_hr.data, sr_path, 'out_hr_{:04d}.png'.format(iter))
                utils.tensor2png(out_lr.data, lr_path, 'out_lr_{:04d}.png'.format(iter))
            
            # adding noise after each training set
            if reg_noise_std > 0:
                if use_cuda:
                    net_input  += (net_input.normal_() * reg_noise_std).cuda()
                else:
                    net_input  += (net_input.normal_() * reg_noise_std)

    #clean up any mess we're leaving on the gpu
    if use_cuda:
        torch.cuda.empty_cache()
