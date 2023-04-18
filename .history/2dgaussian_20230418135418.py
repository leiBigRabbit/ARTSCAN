import torch
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:# 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致        
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8     
        center = ksize // 2    
        xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离    
        ys = (np.arange(ksize, dtype=np.float32) - center)
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernel = kernel1d[..., None] @ kernel1d[None, ...]     
        kernel = torch.from_numpy(kernel)    
        kernel = kernel / kernel.sum() # 归一化return kernel


def GaussianBlur(batch_img, ksize, sigma=None):    
    kernel = getGaussianKernel(ksize, sigma) # 生成权重积核    
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)    
    pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')    
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,                            
                            stride=1, padding=0, groups=C)
    return weighted_pix
