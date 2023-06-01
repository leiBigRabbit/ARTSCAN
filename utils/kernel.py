import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt


def gaussianKernel(ksize, sigma, c):
    center = ksize // 2 
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离 
    kernel1d = np.sqrt(c) * np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
    kernel2d = kernel1d[..., None] @ kernel1d[None, ...]   #@矩阵乘法 得到2维卷积核
    kernel2d = torch.from_numpy(kernel2d)    
    kernel2d = (kernel2d / kernel2d.sum()).to("cuda") # 归一化

    return kernel2d

def getGaussianKernel(ksizes, sigmac, sigmas):
    # if sigma <= 0:# 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致        
    kernel_c = []
    kernel_s = []
    # kernel_G = []
    for ksize, sigc, sigs in zip(ksizes, sigmac, sigmas):
        center = ksize // 2    
        xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离 
        # #Gig
        # kernel1G = np.exp(-(xs ** 2) / (2 * sigG ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        # kernelG = 1 + kernel1G[..., None] @ kernel1G[None, ...]   #@矩阵乘法 得到2维卷积核
        # kernelG = torch.from_numpy(kernelG)    
        # kernel_G.append(kernelG / kernelG.sum()) # 归一化

        #Dc
        kernel1dc = np.exp(-(xs ** 2) / (2 * sigc ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernelc = kernel1dc[..., None] @ kernel1dc[None, ...]   #@矩阵乘法 得到2维卷积核
        kernelc = torch.from_numpy(kernelc)    
        kernel_c.append((kernelc / kernelc.sum()).to("cuda")) # 归一化

        #Ds
        kernel1ds = np.exp(-(xs ** 2) / (2 * sigs ** 2))
        kernels = kernel1ds[..., None] @ kernel1ds[None, ...] 
        kernels = torch.from_numpy(kernels)    
        kernel_s.append((kernels / kernels.sum()).to("cuda")) 
    # i = 0
    # for temps in range(len(kernel_c)):
    #     # for temp in range(len(Gabor_kernels[temps])): 
    #         i += 1
    #         plt.subplot(4, 3, i)   #plt.subplot(nrows, ncols, index)
    #         plt.imshow(kernel_c[temps] - kernel_s[temps])
    #         # i += 1
            # plt.imshow(kernel_c[temps])
    return kernel_c, kernel_s

def Gabor_filter(K_size, sigmav, sigmah, Lambda, angles):  #K_size=111, Sigmav=10, Sigmah=10, Lambda=10, angle=0
    Gabor_kernels = []
    for angle in angles:
        Gabor_kernel_angle = []

        for sigv, sigh, lam, ksize in zip(sigmav, sigmah, Lambda, K_size):
        # get half size
            d_x = ksize[1] // 2
            d_y = ksize[0] // 2
            # prepare kernel
            gabor = np.zeros((ksize[0], ksize[1]), dtype=np.float32)
            # each value
            for y in range(ksize[0]):
                for x in range(ksize[1]):
                    # distance from center
                    px = x - d_x
                    py = y - d_y
        
                    # degree -> radian
                    theta = angle / 180. * np.pi
        
                    #A11 get kernel x
                    _x = np.cos(theta) * px + np.sin(theta) * py
        
                    #A12 get kernel y
                    _y = -np.sin(theta) * px + np.cos(theta) * py

                    #A10 fill kernel
                    # print(np.cos(2*np.pi*_x/lam))
                    gabor[y, x] = 1/(2*np.pi * sigh * sigv) * np.exp(-0.5 * ((_x**2/sigh**2) + (_y**2/sigv**2))) * np.cos(2*np.pi*_x/lam)
                    # gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)
        
            # kernel normalization
            gabor = torch.from_numpy(gabor)
            # gabor_pad[:,7:12] =  gabor[:,7:12]
            # cut = sum(sum(gabor)) - sum(sum(gabor_pad))
            # gabor /= np.sum(np.abs(gabor))
            Gabor_kernel_angle.append((gabor / gabor.sum()).to("cuda"))
        Gabor_kernels.append(Gabor_kernel_angle)
    #画出卷积核
    # i = 0
    # for temps in range(len(Gabor_kernels)):
    #     for temp in range(len(Gabor_kernels[temps])): 
    #         i += 1
    #         plt.subplot(4, 3, i)   #plt.subplot(nrows, ncols, index)
    #         plt.imshow(Gabor_kernels[temps][temp])
    return Gabor_kernels
