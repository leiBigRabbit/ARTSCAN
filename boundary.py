import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

#A24  错误
def Boundary_diffusion(Boundaries):
    Boundary_diffusionP = []
    for Boundary in Boundaries:
        weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0],[1.0, 8.0, 1.0],[1.0, 1.0, 1.0]])) # 根据公式自定义的权值，领域相加，即附近的值*1后与中间值相加，共8个领域，中间值加8次即*8
        B, C, H, W = Boundary.shape
        weight = weight.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        Boundary = F.pad(Boundary, pad=[1, 1, 1, 1], mode='constant') 
        weight = F.conv2d(Boundary, weight=weight, bias=None, stride=1, padding=0, groups=Boundary.shape[0])
        Boundary_diffusionP.append(10**4 /(1+ 40 * weight))
    return Boundary_diffusionP
