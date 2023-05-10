import numpy as np
import torch
from utils.utils import half_rectified
from utils.kernel import Gabor_filter, gaussianKernel, getGaussianKernel
from utils.conv2d import gaussianconv_2d, Gabor_conv, GaussianBlur

#A47
def habituative_transmitter(dt, K_e, Y_ij, C_ij, inter_value):
    Y_ij = Y_ij + dt * (K_e * (2 - Y_ij - 3 * 10 ** 6 * Y_ij * (half_rectified(C_ij) + 625 * inter_value)))
    return Y_ij

#A43
def  Eye_movements_map(dt, E_ij, C_ij, Y_ij, M=0):
    K_kernel = gaussianKernel(3, 5, 1/(50*np.pi))
    J_kernel = gaussianKernel(3, 1, 1/(2*np.pi))
    #A45
    inter_value_J = gaussianconv_2d(E_ij**2, J_kernel, 3)
    #A46
    inter_value_K = gaussianconv_2d(E_ij**2, K_kernel, 3)
    #A47
    y_ij = habituative_transmitter(dt, 10**-7, Y_ij, C_ij, inter_value_J)
    Eij = y_ij * (1 - E_ij) * (half_rectified(C_ij) + 625 * inter_value_J + M) 
    Eij -= E_ij
    Eij_ = - 0.01 * E_ij * sum(half_rectified(C_ij) + inter_value_K)
    Eij = Eij + Eij_
    Eij =  Eij* dt + E_ij
    EIJ = torch.max(Eij)
    output = torch.zeros_like(Eij)
    if EIJ>0.58:
        max_place = (Eij==torch.max(Eij)).nonzero()[0]
        output[max_place[0]][max_place[1]][max_place[2]][max_place[3]] = EIJ
    # Eij = torch.where(Eij >= 0.58, Eij, 0.0)  
    return output, Eij, y_ij
