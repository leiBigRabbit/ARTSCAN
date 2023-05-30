import torch
import cv2
import numpy as np
import torch.nn.functional as F

def write_img(input, name, devide):
    name = "/Users/leilei/Desktop/artimg/" + name
    if type(input) == torch.Tensor:
        if devide:
            try:
                output = cv2.imwrite(name + ".tiff",np.array((input.float()/torch.max(input)).squeeze(dim=0).squeeze(dim=0)))
            except:
                output = cv2.imwrite(name + ".tiff",(input.float()/torch.max(input)).squeeze(dim=0).squeeze(dim=0).detach().numpy())
                
        else:
            output = cv2.imwrite(name + ".tiff",np.array(input.float().squeeze(dim=0).squeeze(dim=0)))
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

def data_process(url):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1000, 1000))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0).unsqueeze(dim=1).float()
    return img