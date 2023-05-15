import torch
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import math
from utils.conv2d import gaussianconv_2d, Gabor_conv, GaussianBlur
from utils.kernel import Gabor_filter, gaussianKernel, getGaussianKernel
from utils.utils import half_rectified, type_input, data_process
from eye_movements import Eye_movements_map
from Spatial_attentional import attention_shroud
from gain import Gain_field
from what_stream import Vector_Bq
from fuzzy_ART import fuzzy_ART
#A6 surface contours
def Surface_contours(Object_surface_ons, Object_surface_offs):
    Jpq_on = 0.8 * Object_surface_ons[0] + 0.1 * Object_surface_ons[1] + 0.1 * Object_surface_ons[2]
    Jpq_off = 0.8 * Object_surface_offs[0] + 0.1 * Object_surface_offs[1] + 0.1 * Object_surface_offs[2]
    K_on_kernel = gaussianKernel(ksize=5, sigma=1, c=1/(2*np.pi))
    K_off_kernel = gaussianKernel(ksize=5, sigma=3, c=1/(2*np.pi*9))
    J = Jpq_on - Jpq_off
    Kernel_add = K_on_kernel + K_off_kernel
    Kernel_cut = K_on_kernel - K_off_kernel
    Kernel_cut1 = K_off_kernel - K_on_kernel
    # plt.subplot(4, 3, 1)   #plt.subplot(nrows, ncols, index)
    # plt.imshow(Kernel_add)
    # plt.imshow(Kernel_cut)
    # plt.imshow(Kernel_cut1)
    # plt.imshow(Kernel_add)
    molecular = gaussianconv_2d(J, Kernel_cut, 5)
    # type_input(molecular, "molecular", 1)
    denominator = 40 + gaussianconv_2d(J, Kernel_add, 5)
    # type_input(denominator, "denominator", 1)
    molecular1 = gaussianconv_2d(J, Kernel_cut1, 5)
    # type_input(molecular1, "molecular1", 1)
    output_C = half_rectified(molecular / denominator) + half_rectified(molecular1 / denominator)
    return output_C

#A18
def Boundaries(Z_complex_cells, C_Surface_contours, M=0):
    output = []
    for i in range(len(Z_complex_cells)):
        Gs_F = gaussianKernel(ksize=5, sigma=5, c=1/(2*np.pi * 25))
        C_S2d = gaussianconv_2d(C_Surface_contours, Gs_F, 5) 
        if i == 2:
            B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d + M) + 0.4 * C_Surface_contours.sum()
            B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d + M)  #- 0.4 * C_Surface_contours.sum()
            B = half_rectified(B_molecular/B_denominator)
            output.append(B)
        else:
            B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d) + 0.4 * C_Surface_contours.sum()
            B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d) - 0.4 * C_Surface_contours.sum()
            B = half_rectified(B_molecular/B_denominator)
            output.append(B)
    return output

#A24中四个领域相加。
def boundary_gated_diffusion(Boundaries):
    boundary_gated_diffusion_P = []
    for Boundary in Boundaries:
        Boundariespg = Boundary.squeeze(0).squeeze(0)  #[:-1, :]
        p = torch.zeros(Boundariespg.shape[1]).unsqueeze(0)
        Boundariespg = torch.cat((p, Boundariespg, p), 0)
        p1 = torch.zeros(Boundariespg.shape[0]).unsqueeze(1)
        Boundariespg = torch.cat((p1, Boundariespg, p1), 1)
        B_up, B_left, B_under, B_right = Boundariespg[:-2, 1:-1].unsqueeze(0).unsqueeze(1), Boundariespg[1:-1, :-2].unsqueeze(0).unsqueeze(1), Boundariespg[2:,1:-1].unsqueeze(0).unsqueeze(1), Boundariespg[1:-1, 2:].unsqueeze(0).unsqueeze(1)
        diffusion_P_up, diffusion_P_under = 10**4 / (1 + 40 * (B_up + Boundary)), 10**4 / (1 + 40 * (B_under + Boundary)),
        diffusion_P_left, diffusion_P_right = 10**4 / (1 + 40 * (B_left + Boundary)), 10**4 / (1 + 40 * (B_right + Boundary)), 
        boundary_gated_diffusion_P.append([diffusion_P_up, diffusion_P_left, diffusion_P_under, diffusion_P_right])
        # boundary_gated_diffusion_P.append(boundary_gated_diffusion_P)
    # type_input(boundary_gated_diffusion_P, "boundary_gated_diffusion_P", 1)
    return boundary_gated_diffusion_P

#A22 A23
def S_reduce_with_P(Surface_filling, boundary_gated_diffusion_P):
    output = []
    for Surface_filling_S in Surface_filling:
        Surface_filling_Spg = Surface_filling_S.squeeze(0).squeeze(0)  #[:-1, :]
        p = torch.zeros(Surface_filling_Spg.shape[1]).unsqueeze(0)
        Surface_filling_Spg = torch.cat((p, Surface_filling_Spg, p), 0)
        p1 = torch.zeros(Surface_filling_Spg.shape[0]).unsqueeze(1)
        Surface_filling_Spg = torch.cat((p1, Surface_filling_Spg, p1), 1)
        B_up, B_left, B_under, B_right = Surface_filling_Spg[:-2, 1:-1], Surface_filling_Spg[1:-1, :-2], Surface_filling_Spg[2:,1:-1], Surface_filling_Spg[1:-1, 2:]
        surface_activity = [B_up - Surface_filling_S, B_left - Surface_filling_S, B_under - Surface_filling_S, B_right - Surface_filling_S]
        output.append(surface_activity)
    outputs = []
    i = 0
    for surface, boundary in zip(output, boundary_gated_diffusion_P):
        x = []
        for i in range(len(surface)):
            x.append(surface[i] * boundary[i])
        outputs.append(sum(x))
    type_input(outputs, "sp", 1)
    return outputs

#A22
def Surface_filling_in_(dt, Boundary, S_filling_ins, output_signal_sf, X_input, Rwhere=0):
    Pboundary_gated_diffusion = boundary_gated_diffusion(Boundary)
    s_with_p = S_reduce_with_P(S_filling_ins, Pboundary_gated_diffusion)

    output = []
    for i  in range(len(S_filling_ins)):
        S_filling_in = S_filling_ins[i]
        #A22
        output.append((S_filling_in + (-80) * S_filling_in + s_with_p[i] + 100 * X_input[i] * (1 + output_signal_sf) - S_filling_in * Rwhere) * dt)
    return output

#A22
def Surface_filling_in(dt, Boundary_diffusionP, S_filling_ins, output_signal_sf, X_input, Rwhere=0):
    output = []
    for i  in range(len(S_filling_ins)):
        S_filling_in = S_filling_ins[i]
        B, C, H, W = S_filling_in.shape
        #A25
        weight = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0],[1.0, -8.0, 1.0],[1.0, 1.0, 1.0]]))
        weight = weight.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        S_filling_in_conv = F.pad(S_filling_in, pad=[1, 1, 1, 1], mode='constant') 
        S_filling_in_conv = F.conv2d(S_filling_in_conv, weight=weight, bias=None, stride=1, padding=0, groups=S_filling_in.shape[0])
        #A22
        output.append((S_filling_in + (-80) * S_filling_in + Boundary_diffusionP[i] * S_filling_in_conv + 100 * X_input[i] * (1 + output_signal_sf) - S_filling_in * Rwhere) * dt)
    return output

#A16
def complex_cells(Y_ons, Y_offs):
    #A15
    Y_add = np.array(Y_ons).sum(axis=0)
    Y_cuts = np.array(Y_offs).sum(axis=0)
    input = np.array([Y_add, Y_cuts]).sum(axis=0)
    # type_input(input, "output_on1", 1)
    #A17
    L_gskernel = gaussianKernel(3, 1, 1/(2*np.pi))
    output = []
    for Z in input:
        complex = half_rectified(Z ** 2 / (0.01 + gaussianconv_2d(Z**2, L_gskernel, (3,3))))
        output.append(complex)
    return output

def main():
    global dt
    dt = 0.05
    img = data_process("/Users/leilei/Desktop/ARTSCAN/image.jpg")
    #A.1. Retina and LGN cells
    imgsc_on, imgsc_off = GaussianBlur(img, [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7])
    #A.2. V1 polarity-sensitive oriented simple cells
    #A13
    Y_ons, Y_offs = Gabor_conv(imgsc_on, imgsc_off, sigmav= [3, 4.5, 6], sigmah = [1, 1.5, 2], Lambda = [3, 5, 7], angles = [0, 45, 90, 135], K_size = [(19,5), (29,7), (39,9)])
    # type_input(Y_offs, "Y_offs", 1)
    #A16
    Z = complex_cells(Y_ons, Y_offs)
    # type_input(Z, "output_onZ", 1)

    for t in range(1,40):
        
        # type_input(Binit, "output_onZ", 1)
        if t == 1:
            Object_surface_ons = [torch.zeros(Z[0].shape), torch.zeros(Z[0].shape), torch.zeros(Z[0].shape)]
            Object_surface_offs = [torch.zeros(Z[0].shape), torch.zeros(Z[0].shape), torch.zeros(Z[0].shape)]
            #A27
            Cij = Surface_contours(Object_surface_ons, Object_surface_offs)
            # type_input(Cij, "Cij", 1) 
            Eij, Y_ij = torch.zeros(Z[0].shape), torch.zeros(Z[0].shape)
            Amn = torch.zeros(Z[0].shape)
        M = 0
        Boundary = Boundaries(Z, Cij, M)
        type_input(Boundary, "Boundary", 1) 
        Bq = Vector_Bq(Boundary[2])
        
        model = fuzzy_ART(X_size=100 * 100, c_max=100, rho=0.85, alpha=0.00001, beta=1)
        _,_,W = model.train_batch(Bq)
        
        
        # type_input(Boundary, "Boundary", 1) 
        #A24
        # Pboundary_gated_diffusion = Boundary_diffusion(Binit)
        #A22
        #A23
        if t < 20:
            output_signal_sf = 0 
            Rwhere=0
        Object_surface_ons = Surface_filling_in_(dt, Boundary, Object_surface_ons, output_signal_sf,  imgsc_on, Rwhere)
        Object_surface_offs = Surface_filling_in_(dt, Boundary, Object_surface_offs, output_signal_sf,  imgsc_off)

        # Object_surface_ons = Surface_filling_in(dt, Pboundary_gated_diffusion, Object_surface_ons, output_signal_sf,  imgsc_on)
        #A23
        # Object_surface_offs = Surface_filling_in(dt, Pboundary_gated_diffusion, Object_surface_offs, output_signal_sf, imgsc_off)
        #A26
        S_ij = 0.05 * (Object_surface_ons[0] + Object_surface_offs[0]) + 0.1 * (Object_surface_ons[1] + Object_surface_offs[1]) + 0.85 * (Object_surface_ons[2] + Object_surface_offs[2])
        # type_input(S_ij, "S_ij", 1)
        Cij = Surface_contours(Object_surface_ons, Object_surface_offs)
        type_input(Cij, str(t), 1) 
        #A43
        EIJ, Eij, Y_ij, max_place = Eye_movements_map(dt, Eij, Cij, Y_ij, M=0)
        #A34  max_place, Amn
        #临时定义
        max_place = [0,0,50,150]
        SFmn, AImn = Gain_field(S_ij, max_place, Amn)


if __name__ == "__main__":
    main()

