import torch
import numpy as np
import torch.nn.functional as F
import cv2

@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:# 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致        
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8     
        center = ksize // 2    
        xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离    
        # ys = (np.arange(ksize, dtype=np.float32) - center)
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) #+ ys ** 2 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernel = kernel1d[..., None] @ kernel1d[None, ...]     
        kernel = torch.from_numpy(kernel)    
        kernel = kernel / kernel.sum() # 归一化return kernel
    return kernel


def GaussianBlur(batch_img, ksize, sigma=None):    
    kernel = getGaussianKernel(ksize, sigma) # 生成权重积核    
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)    
    pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')    
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,                            
                            stride=1, padding=0, groups=C)
    return weighted_pix


def CVGaussianBlur(batch_img, ksize):
    
    weighted_pixs = []
    for ks in ksize:
        sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
        kernel_1d = cv2.getGaussianKernel(ksize=ks, sigma=sigma, ktype=cv2.CV_32F)
        kernel_2d = kernel_1d * kernel_1d.T
        kernel = torch.from_numpy(kernel_2d)
        B, C, H, W = batch_img.shape
        kernel = kernel.view(1, 1, ks, ks).repeat(C, 1, 1, 1)    
        pad = (ks - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant')    
        weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,                            
                                stride=1, padding=0, groups=C)
        weighted_pixs.append(np.array(weighted_pix.squeeze(dim=0)))
    
    return weighted_pixs


if __name__ == "__main__":
    img = cv2.imread('image.jpg')
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('output.jpg', gray_img)
    img1 = GaussianBlur(img.float(), 5, 0)
    img1 = img1.squeeze(dim=0)
    img1 = np.array(img1)
    cv2.imwrite("output2.jpg", img1)

    imgs = CVGaussianBlur(img.float(), [17])

    cv2.imwrite("output3.jpg", img2)


    


