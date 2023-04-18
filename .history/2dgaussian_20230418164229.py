import torch
import numpy as np
import torch.nn.functional as F
import cv2

@torch.no_grad()
def getGaussianKernel(ksizes, sigmac, sigmas):
    # if sigma <= 0:# 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致        
    kernel_c = []
    kernel_s = []
    for ksize, sigc, sigs in zip(ksizes, sigmac, sigmas):

        center = ksize // 2    
        xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离 
        #Dc
        kernel1dc = np.exp(-(xs ** 2) / (2 * sigc ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernelc = kernel1dc[..., None] @ kernel1dc[None, ...]   #@矩阵乘法 得到2维卷积核
        kernelc = torch.from_numpy(kernelc)    
        kernel_c.append(kernelc / kernelc.sum()) # 归一化return kernel
        #Ds
        kernel1ds = np.exp(-(xs ** 2) / (2 * sigs ** 2))
        kernels = kernel1ds[..., None] @ kernel1ds[None, ...] 
        kernels = torch.from_numpy(kernels)    
        kernel_s.append(kernels / kernels.sum()) 
    return kernel_c, kernel_s


def GaussianBlur(batch_img, ksize, sigmac, sigmas):    
    kernel_c, kernel_s = getGaussianKernel(ksize, sigmac, sigmas) # 生成权重积核    
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    

    weighted_pixs = []
    for ks, kern_c, kern_s in zip(ksize, kernel_c, kernel_s):
        kern_c = kern_c.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        kern_s = kern_s.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        pad = (ks - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant')    

        weighted_Dcg = F.conv2d(batch_img_pad, weight=kern_c, bias=None, stride=1, padding=0, groups=C)
        weighted_Dcs = F.conv2d(batch_img_pad, weight=kern_s, bias=None, stride=1, padding=0, groups=C)

        weighted_pix = (weighted_Dcg - weighted_Dcs)/(1 + weighted_Dcg + weighted_Dcs)
        weighted_pixs.append(weighted_pix)
    return weighted_Dcg, weighted_Dcs, weighted_pixs


# def GaussianBlur(batch_img, ksize, sigmac, sigmas):    
#     kernels = getGaussianKernel(ksize, sigmac, sigmas) # 生成权重积核    
#     B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    
#     kernel = [kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1) for kernel in kernels]  
#     pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
#     batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant')    
#     weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,                            
#                             stride=1, padding=0, groups=C)
#     return weighted_pix


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


if __name__ == "__main__":
    img = cv2.imread('image.jpg')
    # img = cv2.resize(img, (640, 640))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('output.jpg', gray_img)
    imgsc, imgss, imgsc_s = GaussianBlur(img.float(), [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7])
    # imgs = imgs[0].squeeze(dim=0)
    # img1 = np.array(imgs)
    cv2.imwrite("output2.jpg",np.array(imgsc[0].squeeze(dim=0)))
    cv2.imwrite("output3.jpg",np.array(imgss[0].squeeze(dim=0)))
    cv2.imwrite("output4.jpg",np.array(imgsc_s[0].squeeze(dim=0)))

    # imgs = CVGaussianBlur(img.float(), [5, 17, 41])
    # cv2.imwrite("output3.jpg", imgs[0])
    # cv2.imwrite("output4.jpg", imgs[1])
    # cv2.imwrite("output5.jpg", imgs[2])


    


