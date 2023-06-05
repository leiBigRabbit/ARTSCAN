import torch
import cv2
import numpy as np
import torch.nn.functional as F
import os
from utils.kernel import Gabor_filter, getGaussianKernel, gaussianKernel
from utils.conv2d import gaussianconv_2d



#A24中四个领域相加。
def boundary_gated_diffusion(Boundaries):
    boundary_gated_diffusion_P = []
    for Boundary in Boundaries:
        Boundariespg = Boundary.squeeze(0).squeeze(0)  #[:-1, :]
        p = torch.zeros(Boundariespg.shape[1]).unsqueeze(0).cuda()
        Boundariespg = torch.cat((p, Boundariespg, p), 0)
        p1 = torch.zeros(Boundariespg.shape[0]).unsqueeze(1).cuda()
        Boundariespg = torch.cat((p1, Boundariespg, p1), 1)
        B_up, B_left, B_under, B_right = Boundariespg[:-2, 1:-1].unsqueeze(0).unsqueeze(1), Boundariespg[1:-1, :-2].unsqueeze(0).unsqueeze(1), Boundariespg[2:,1:-1].unsqueeze(0).unsqueeze(1), Boundariespg[1:-1, 2:].unsqueeze(0).unsqueeze(1)
        diffusion_P_up, diffusion_P_under = 10 / (1 + 40 * (B_up + Boundary)), 10 / (1 + 40 * (B_under + Boundary)),
        diffusion_P_left, diffusion_P_right = 10 / (1 + 40 * (B_left + Boundary)), 10 / (1 + 40 * (B_right + Boundary)), 
        boundary_gated_diffusion_P.append([diffusion_P_up, diffusion_P_left, diffusion_P_under, diffusion_P_right])
        # boundary_gated_diffusion_P.append(boundary_gated_diffusion_P)
    # type_input(boundary_gated_diffusion_P, "boundary_gated_diffusion_P", 1)
    return boundary_gated_diffusion_P

#A22 A23
def S_reduce_with_P(Surface_filling, boundary_gated_diffusion_P):
    output = []
    for Surface_filling_S in Surface_filling:
        Surface_filling_Spg = Surface_filling_S.squeeze(0).squeeze(0)  #[:-1, :]
        p = torch.zeros(Surface_filling_Spg.shape[1]).unsqueeze(0).cuda()
        Surface_filling_Spg = torch.cat((p, Surface_filling_Spg, p), 0)
        p1 = torch.zeros(Surface_filling_Spg.shape[0]).unsqueeze(1).cuda()
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
    # type_input(outputs, "sp", 1)
    return outputs

#A16
def complex_cells(Y_ons, Y_offs):
    Y_add = np.array([[item.cpu().detach().numpy() for item in Y_ons[i]] for i in range(len(Y_ons))]).sum(axis=0)
    Y_cuts = np.array([[item.cpu().detach().numpy() for item in Y_offs[i]] for i in range(len(Y_offs))]).sum(axis=0)
    #A15
    # Y_add = np.array(Y_ons).sum(axis=0)
    # Y_cuts = np.array(Y_offs).sum(axis=0)
    input = np.array([Y_add, Y_cuts]).sum(axis=0)
    # type_input(input, "output_on1", 1)
    #A17
    L_gskernel = gaussianKernel(5, 1, 1/(2 * np.pi))
    output = []
    for Z in input:
        complex = F.relu(torch.tensor(Z).cuda() ** 2 / (0.01 + gaussianconv_2d(torch.tensor(Z).cuda()**2, L_gskernel, (5,5))))
        output.append(complex)
    return output



def write_img(input, name, devide):
    if not os.path.exists("./artimg"):
        os.mkdir("./artimg")
    name = "./artimg/" + name
    if type(input) == torch.Tensor:
        if devide:
            try:
                output = cv2.imwrite(name + ".tiff",np.array((input.cpu().float()/torch.max(input)).squeeze(dim=0).squeeze(dim=0)))
            except:
                output = cv2.imwrite(name + ".tiff",(input.cpu().float()/torch.max(input)).squeeze(dim=0).squeeze(dim=0).detach().numpy())
                
        else:
            output = cv2.imwrite(name + ".tiff",np.array(input.cpu().float().squeeze(dim=0).squeeze(dim=0)))
    return output

def type_input(inputs, name, devide):
    if type(inputs) == torch.Tensor:
        write_img(inputs, name, devide)
    elif type(inputs) == tuple() or type(inputs) == list or type(inputs) == np.ndarray:
        i = 0
        for input in inputs:
            i += 1
            if type(input) == torch.Tensor:
                write_img(input, name + str(i), devide)
            
            elif type(input) == list:
                j = 0
                for inp in input:
                    j += 1
                    write_img(inp, name + '_' + str(i) + '_' + str(j), devide)
    return 0


def  half_rectified(input, type = 1):
    if type == 0:
        output = torch.where(input < 0, 0 * input, input)
    elif type == 1:
        output = F.relu(input)
    return output

def half_rectified_relu(input):
    output = F.relu(input)
    return output

#A38
def  signal_g_function(input):
    output = half_rectified(input - 0.05)
    return output

#A21
def signal_m_function(input):
    input = F.relu(input)
    output = input / (0.001 + input)
    return output

#A39
def signal_f_function(input):
    output = 4 * input ** 4 / (0.35**4 + input ** 4)
    return output



#A51 
def signal_k_function(input):
    output = half_rectified(input ** 35) / (0.22**35 + half_rectified(input*35))
    return output

#A52
def neurotransmitter_yR(input, r_where, dt=0.05):
    output = 6 * (2 - input, -2 * input * r_where) * dt + input
    return output

def data_process(url, size):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0).unsqueeze(dim=1).float()
    return img