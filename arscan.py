import torch
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image

@torch.no_grad()
def Gabor_filter(K_size, sigmav, sigmah, Lambda, angle):  #K_size=111, Sigmav=10, Sigmah=10, Lambda=10, angle=0
    Gabor_kernels = []
    for sigv, sigh, lam, ksize in zip(sigmav, sigmah, Lambda, K_size):
    # get half size
        d_y = ksize[0] // 2
        d_x = ksize[1] // 2
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
                gabor[y, x] = 1/(2*np.pi * sigh * sigv) * np.exp(-0.5 * ((_x**2/sigh**2) + (_y**2/sigv**2))) * np.cos(2*np.pi*_x/lam)
                # gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)
    
        # kernel normalization
        gabor /= np.sum(np.abs(gabor))
        Gabor_kernels.append(torch.from_numpy(gabor))
    return Gabor_kernels

def getGaussianKernel(ksizes, sigmac, sigmas, sigmaG):
    # if sigma <= 0:# 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致        
    kernel_c = []
    kernel_s = []
    kernel_G = []
    for ksize, sigc, sigs, sigG in zip(ksizes, sigmac, sigmas, sigmaG):
        center = ksize // 2    
        xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离 
        #Gig
        kernel1G = np.exp(-(xs ** 2) / (2 * sigG ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernelG = 1 + kernel1G[..., None] @ kernel1G[None, ...]   #@矩阵乘法 得到2维卷积核
        kernelG = torch.from_numpy(kernelG)    
        kernel_G.append(kernelG / kernelG.sum()) # 归一化

        #Dc
        kernel1dc = np.exp(-(xs ** 2) / (2 * sigc ** 2)) # 计算一维卷积核# 根据指数函数性质，利用矩阵乘法快速计算二维卷积核    
        kernelc = kernel1dc[..., None] @ kernel1dc[None, ...]   #@矩阵乘法 得到2维卷积核
        kernelc = torch.from_numpy(kernelc)    
        kernel_c.append(kernelc / kernelc.sum()) # 归一化

        #Ds
        kernel1ds = np.exp(-(xs ** 2) / (2 * sigs ** 2))
        kernels = kernel1ds[..., None] @ kernel1ds[None, ...] 
        kernels = torch.from_numpy(kernels)    
        kernel_s.append(kernels / kernels.sum()) 
    return kernel_c, kernel_s, kernel_G

def Gabor_conv(imgsc_on, imgsc_off, sigmav, sigmah, Lambda, angle, K_size):
    Gabor_filter_list = Gabor_filter(K_size, sigmav, sigmah, Lambda, angle)

    Y_ons = []
    Y_offs = []
    for kern_gab, ks, im_on, im_off in zip(Gabor_filter_list, K_size, imgsc_on, imgsc_off):
        kern_gab = kern_gab.view(1, 1, ks[0], ks[1]).repeat(im_on.shape[0], 1, 1, 1)
        pad_y = (ks[0] - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
        pad_x = (ks[1] - 1) // 2
        im_on = F.pad(im_on, pad=[pad_x, pad_x, pad_y, pad_y], mode='constant')    

        weighted_on = F.conv2d(im_on, weight=kern_gab, bias=None, stride=1, padding=0, groups=im_on.shape[0])
        weighted_off = F.conv2d(im_off, weight=kern_gab, bias=None, stride=1, padding=0, groups=im_off.shape[0])
        cv2.imwrite("output7_1.tiff",np.array(weighted_on.float().squeeze(dim=0).squeeze(dim=0)))
        cv2.imwrite("output7_2.tiff",np.array(weighted_off.float().squeeze(dim=0).squeeze(dim=0)))
        exit()
        #A9
        Y_on = weighted_on - weighted_off
        Y_off = - Y_on

        #A13
        Y_on = torch.where(Y_on < 0, 0, Y_on)
        Y_ons.append(Y_on)

        #A14
        Y_off = torch.where(Y_off < 0, 0, Y_off)
        Y_offs.append(Y_off)

    return Y_ons, Y_offs

def GaussianBlur(batch_img, ksize, sigmac, sigmas, sigmaG):    
    kernel_c, kernel_s, kernel_G = getGaussianKernel(ksize, sigmac, sigmas, sigmaG) # 生成权重积核    
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到# 生成 group convolution 的卷积核    

    weighted_pixs_on = []
    weighted_pixs_off = []
    weighted_c_s = []
    weighted_c__s = []

    for ks, kern_c, kern_s, kern_g in zip(ksize, kernel_c, kernel_s, kernel_G):
        kern_c = kern_c.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        kern_s = kern_s.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
        kern_g = kern_g.view(1, 1, ks, ks).repeat(C, 1, 1, 1)

        pad = (ks - 1) // 2 # 保持卷积前后图像尺寸不变# mode=relfect 更适合计算边缘像素的权重    
        batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant')    

        weighted_Dcg = F.conv2d(batch_img_pad, weight=kern_c, bias=None, stride=1, padding=0, groups=C)
        weighted_Dcs = F.conv2d(batch_img_pad, weight=kern_s, bias=None, stride=1, padding=0, groups=C)
        
        #中间过程
        weighted_c_s.append(weighted_Dcg.float() - weighted_Dcs.float())
        weighted_c__s.append(1 + weighted_Dcg.float() + weighted_Dcs.float())

        #on and off cell
        weighted_pix_on = (weighted_Dcg.float() - weighted_Dcs.float())/(1 + weighted_Dcg.float() + weighted_Dcs.float()).float()
        weighted_pix_off = -weighted_pix_on
        
        #on cell
        weighted_pix_on = torch.where(weighted_pix_on < 0, 0, weighted_pix_on)
        weighted_pix_on = F.conv2d(weighted_pix_on, weight=kern_g, bias=None, stride=1, padding=0, groups=C)
        weighted_pixs_on.append(weighted_pix_on)
        
        #off cell
        weighted_pix_off = torch.where(weighted_pix_off < 0, 0, weighted_pix_off)
        weighted_pix_off = F.conv2d(weighted_pix_off, weight=kern_g, bias=None, stride=1, padding=0, groups=C)
        weighted_pixs_off.append(weighted_pix_off)

    return weighted_Dcg, weighted_Dcs, weighted_pixs_on, weighted_pixs_off, weighted_c_s, weighted_c__s


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
    img = cv2.imread('image1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (640, 640))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)
    img = img.unsqueeze(dim=1)
    #imgsc 是高斯dc卷积后的结果
    #imgss 是高斯ds卷积后的结果
    #imgsc_s是相减相加后的结果
    #imgsccuts, imgscadds是中间过程
    imgsc, imgss, imgsc_on, imgsc_off, imgsccuts, imgscadds = GaussianBlur(img.float(), [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7], sigmaG = [6, 6, 6])
    sigmav = [1, 1.5, 2]
    sigmah = [3, 4.5, 6]
    Lambda = [3, 5, 7]
    angle = 45
    K_size = [(19,5), (29,7), (39,9)]
    Y_ons, Y_offs = Gabor_conv(imgsc_on, imgsc_off, sigmav, sigmah, Lambda, angle, K_size)
    cv2.imwrite("output5_1.tiff",np.array(Y_ons[0].float().squeeze(dim=0).squeeze(dim=0)))
    cv2.imwrite("output5_2.tiff",np.array(Y_ons[1].float().squeeze(dim=0).squeeze(dim=0)))
    cv2.imwrite("output5_3.tiff",np.array(Y_ons[2].float().squeeze(dim=0).squeeze(dim=0)))

    cv2.imwrite("output6_1.tiff",np.array(Y_offs[0].float().squeeze(dim=0).squeeze(dim=0)))
    cv2.imwrite("output6_2.tiff",np.array(Y_offs[1].float().squeeze(dim=0).squeeze(dim=0)))
    cv2.imwrite("output6_3.tiff",np.array(Y_offs[2].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output1.jpg",np.array(imgsc[0].squeeze(dim=0)))
    # cv2.imwrite("output1.tiff",np.array(imgsc[0].squeeze(dim=0)/255.0))

    # cv2.imwrite("output3_1.tiff",np.array(imgsc_on[0].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output3_2.tiff",np.array(imgsc_on[1].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output3_3.tiff",np.array(imgsc_on[2].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output4_1.tiff",np.array(imgsc_off[0].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output4_2.tiff",np.array(imgsc_off[1].float().squeeze(dim=0).squeeze(dim=0)))
    # cv2.imwrite("output4_3.tiff",np.array(imgsc_off[2].float().squeeze(dim=0).squeeze(dim=0)))



    

