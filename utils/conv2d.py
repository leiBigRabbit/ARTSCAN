import torch.nn.functional as F
import numpy as np
import torch
import cv2
from .kernel import Gabor_filter, getGaussianKernel, gaussianKernel
import matplotlib.pyplot as plt


def gaussianconv_2d(input, kernel, k_size):
    k_size = kernel.shape
    kernel = kernel.view(1, 1, k_size[0], k_size[1]).repeat(input.shape[0], 1, 1, 1)
    pad_y = (k_size[0] - 1) // 2
    pad_x = (k_size[1] - 1) // 2
    input = F.pad(input, pad=[pad_x, pad_x, pad_y, pad_y], mode='constant') 
    output = F.conv2d(input, weight=kernel, bias=None, stride=1, padding=0, groups=input.shape[0])
    return output

def Gabor_conv(imgsc_on, imgsc_off, sigmav, sigmah, Lambda, angles, K_size):
    Gabor_filter_lists = Gabor_filter(K_size, sigmav, sigmah, Lambda, angles)
    Y_ons_all = []
    Y_offs_all = []
    i = 0
    for Gabor_filter_list in Gabor_filter_lists:
        i +=   1
        Y_ons = []
        Y_offs = []
        for kern_gab, ks, im_on, im_off in zip(Gabor_filter_list, K_size, imgsc_on, imgsc_off):
            kern_gab = kern_gab.view(1, 1, ks[0], ks[1]).repeat(im_on.shape[0], 1, 1, 1)
            pad_y = (ks[0] - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
            pad_x = (ks[1] - 1) // 2
            im_on = F.pad(im_on, pad=[pad_x, pad_x, pad_y, pad_y], mode='constant')  
            # cv2.imwrite("im_on_" + str(i) + ".tiff", np.array((im_on/torch.max(im_on)).squeeze(dim=0).squeeze(dim=0).detach().numpy()))
            im_off = F.pad(im_off, pad=[pad_x, pad_x, pad_y, pad_y], mode='constant') 
            # cv2.imwrite("im_off_" + str(i) + ".tiff", np.array((im_off/torch.max(im_off)).squeeze(dim=0).squeeze(dim=0).detach().numpy()))
            #A9
            weighted_on = F.conv2d(im_on, weight=kern_gab, bias=None, stride=1, padding=0, groups=im_on.shape[0])
            # type_input(weighted_on, "weighted_on", 1)
            
            weighted_off = F.conv2d(im_off, weight=kern_gab, bias=None, stride=1, padding=0, groups=im_off.shape[0])
            #A9    
            Y_on = weighted_on - weighted_off

            Y_off = - Y_on
            #A13
            Y_on = torch.where(Y_on < 0, 0*Y_on, Y_on)
            Y_ons.append(Y_on)
            #A14
            Y_off = torch.where(Y_off < 0, 0*Y_off, Y_off)
            Y_offs.append(Y_off)
        Y_ons_all.append(Y_ons)
        Y_offs_all.append(Y_offs)
    return Y_ons_all, Y_offs_all

def GaussianBlur(batch_img, ksize, sigmac, sigmas):    
    kernel_G = gaussianKernel(500, 6, 1)
    # plt.subplot(4, 3, 1)   #plt.subplot(nrows, ncols, index)
    # plt.imshow(kernel_G+1)
    kernel_c, kernel_s = getGaussianKernel(ksize, sigmac, sigmas) # 生成权重积核
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    
    weighted_pixs_on = []
    weighted_pixs_off = []
    i = 0
    for ks, kern_c, kern_s in zip(ksize, kernel_c, kernel_s):
        i+= 1
        kern_c = kern_c.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        kern_s = kern_s.view(1, 1, ks, ks).repeat(C, 1, 1, 1)

        pad = (ks - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant')    
        #A6分子
        weighted_pix_on_1 = F.conv2d(batch_img_pad, weight=kern_c-kern_s, bias=None, stride=1, padding=0, groups=C)
        #A6分母
        weighted_pix_on_2 = 1 + F.conv2d(batch_img_pad, weight=kern_c+kern_s, bias=None, stride=1, padding=0, groups=C)
        #A6
        weighted_pix_on = weighted_pix_on_1 / weighted_pix_on_2
        weighted_pix_off = -weighted_pix_on
        #on cell
        #A4
        weighted_pix_on = F.relu(weighted_pix_on)   #torch.where(weighted_pix_on < 0, 0, weighted_pix_on)
        weighted_pix_off =F.relu(weighted_pix_off)     #torch.where(weighted_pix_off < 0, 0, weighted_pix_off)

        weighted_pix_on = weighted_pix_on * (1 + kernel_G)
        weighted_pixs_on.append(weighted_pix_on)
        #A5
        # weighted_pix_off = F.pad(weighted_pix_off, pad=[pad, pad, pad, pad], mode='constant')
        # weighted_pix_off = F.conv2d(weighted_pix_off, weight=kern_g, bias=None, stride=1, padding=0, groups=C)
        weighted_pix_off = weighted_pix_off * (1 + kernel_G)
        weighted_pixs_off.append(weighted_pix_off)
    return weighted_pixs_on, weighted_pixs_off


def CVGaussianBlur(batch_img, ksize): 
    weighted_pixs = []
    for ks in ksize:
        sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
        kernel_1d = cv2.getGaussianKernel(ksize=ks, sigma=sigma, ktype=cv2.CV_32F)
        kernel_2d = kernel_1d * kernel_1d.T
        kernel = torch.from_numpy(kernel_2d)
        B, C, H, W = batch_img.shape
        kernel = kernel.view(1, 1, ks, ks).repeat(C, 1, 1, 1)    
        pad =  (ks - 1) // 2 # 保持卷积前后图像尺寸不变# mode=reflect 更适合计算边缘像素的权重    
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='replicate')    
        weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,                            
                                stride=1, padding=0, groups=C)
        weighted_pixs.append(np.array(weighted_pix.squeeze(dim=0)))
    
    return weighted_pixs
