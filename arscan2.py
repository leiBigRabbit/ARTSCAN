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
from utils.utils import half_rectified, type_input, data_process, signal_m_function, half_rectified_relu
from eye_movements import Eye_movements_map
from Spatial_attentional import attention_shroud
from gain import Gain_field
from what_stream import Vector_Bq, activity_V, what_stream
from fuzzy_ART import fuzzy_ART
from View_category_integrator import View_category_integrator
from Invariant_object_category import object_category_neuron

class arcscan():
    def __init__(self, size=(500, 500), dt = 0.05, t=0.6, Rwhere=0, key="up"):
        self.device="cpu"
        self.dt = dt
        self.size = size
        self.batch = 25
        self.size_ = (self.batch, size[0]//5)
        self.Rwhere = 0
        self.category = 100
        self.Object_surface_ons = [torch.zeros(size), torch.zeros(size), torch.zeros(size)]
        self.Object_surface_offs = [torch.zeros(size), torch.zeros(size), torch.zeros(size)]
        self.Eij = torch.zeros(size)
        self.Y_ij = torch.zeros(size)
        self.Amn = torch.zeros(size)
        self.Cij = torch.zeros(1, 1, size[0], size[1])
        self.Sijf = torch.zeros(size)
        self.M = torch.zeros(size)
        self.B = torch.zeros(size)
        self.fuzzy_ART = fuzzy_ART(X_size=100 * 100, c_max=100, rho=0.85, alpha=0.00001, beta=1)
        self.t = t
        self.t_view = [{"Vjiq": torch.zeros( (100) ).to(self.device),
                        "Oj": torch.zeros( (100) ).to(self.device), 
                        "Fj": torch.zeros( (100) ).to(self.device), 
                        "Dj": torch.zeros( (100) ).to(self.device), 
                        "Nj": torch.zeros( (100) ).to(self.device)},
                       {"Vjiq": torch.zeros( (100) ).to(self.device),
                        "Oj": torch.zeros( (100) ).to(self.device), 
                        "Fj": torch.zeros( (100) ).to(self.device), 
                        "Dj": torch.zeros( (100) ).to(self.device), 
                        "Nj": torch.zeros( (100) ).to(self.device)}]

        #up
        self.W_vo = torch.ones( (self.size_) ).to(self.device)
        self.W_od = torch.ones( (self.size_) ).to(self.device)
        self.W_df = torch.ones( (self.size_) ).to(self.device)
        self.W_fn = torch.ones( (self.size_) ).to(self.device)
        self.W_of = torch.ones( (self.size_) ).to(self.device)
        #down
        self.W_nf = torch.ones( (self.size_) ).to(self.device)
        self.W_fo = torch.ones( (100) ).to(self.device)
        self.W_ov = torch.ones( (self.size_) ).to(self.device)

        self.W_ve = torch.ones((self.size_)).to(self.device)  #****

        self.Eij, self.Y_ij = torch.zeros(self.size), torch.zeros(self.size)
        self.Rwhere = Rwhere
        self.key = key
        self.G = self.get_G(self.key)
        self.Uj = 0
        self.Tj = 0
        self.Pj = 0

        self.K_e = 10**-7

    def update(self, value):
        self.t_view[0] = self.t_view[1]
        self.t_view[1] = value

    def part1(self, input):
        Z = input["Z"], 
        imgsc_on =  input["imgsc_on"]
        imgsc_off = input["imgsc_off"]
        self.B = self.Boundaries(Z[0])
        self.Object_surface_ons = self.Surface_filling_in_(self.B, self.Object_surface_ons, imgsc_on)
        self.Object_surface_offs = self.Surface_filling_in_(self.B, self.Object_surface_offs, imgsc_off)
        self.Cij = self.Surface_contours()
        self.S_ij = 0.05 * (self.Object_surface_ons[0] + self.Object_surface_offs[0]) + 0.1 * (self.Object_surface_ons[1] + self.Object_surface_offs[1]) + 0.85 * (self.Object_surface_ons[2] + self.Object_surface_offs[2])
        # return self.B
    #view category integrators
    def part2_up(self, input):
        self.Vjq = self.normalized_V_(input)
        value = self.t_view[0]
        #Vjiq
        Vjiq = (-0.01 * value["Vjiq"] + self.t * ((1 + torch.sum(self.W_ov * signal_m_function(value["Oj"]),dim=1)).unsqueeze(dim=1) * (half_rectified(self.Vjq) + self.G)) - self.Rwhere) * self.dt + value["Vjiq"]
        
        #A63
        Oj = value['Oj'] + self.dt * (-value['Oj'] + (1 + 2 * value['Fj'] * self.W_fo) * (0.5 * torch.sum(signal_m_function(half_rectified_relu(value['Vjiq']))* self.W_vo)  + self.G) - self.Rwhere)
        Oj = self.normalized_O_(Oj)
        
        #A66
        Dj = -value['Dj'] + self.Uj + self.Tj + 0.1 * torch.sum(signal_m_function(value['Oj']) * self.W_od)   #待修改
        Dj = self.normalized_D_(Dj)

        Fj = (-value['Fj'] + (0.5 * signal_m_function(value['Oj']) * self.W_of + self.G) * (1 + torch.sum(value['Dj'] * self.W_df + torch.sum(signal_m_function(value['Nj']) * self.W_nf)))) * self.dt * 20 + value['Fj']
        Fj = self.normalized_F_(Fj)

        Nj = (- value['Nj'] + torch.sum(signal_m_function(value['Fj']) * self.W_fn) + self.Pj) * 20 * self.dt
        Nj = self.normalized_N_(Nj)
        value_dic = {"Vjiq":Vjiq, "Oj":Oj, "Dj":Dj, "Fj":Fj, "Nj":Nj}
        self.update(value_dic)
        
    
        # self.W_ov = (signal_m_function(self.Oj) * F.relu(self.Vjiq) * (F.relu(self.Vjiq) - self.W_ov)) * self.dt + self.W_ov

    def part2_down(self):
        value = self.t_view[1]
        #80
        self.W_fn = signal_m_function(value['Fj']) * (signal_m_function(value['Nj']) - self.W_fn) * self.dt /50 + self.W_fn
        #A76
        self.W_of = (signal_m_function(value['Oj'])* (signal_m_function(value['Fj']) - self.W_of)) * self.dt / 50 + self.W_of
        #A78
        self.W_od = signal_m_function(value['Oj']) * signal_m_function(value['Dj']) * (signal_m_function(value['Dj']) - self.W_od) * self.dt / 50 + self.W_od
        
        self.W_nf = signal_m_function(value['Nj']) * signal_m_function(value['Fj']) *(signal_m_function(value['Fj']) - self.W_nf) * self.dt /50 + self.W_nf
        # self.Fj = (-self.Fj + (0.5 * signal_m_function(self.Oj) * self.W_of + self.G) * (1 + torch.sum(self.Dj * self.W_df + torch.sum(signal_m_function(self.Ni) * self.W_nf)))) * self.dt * 20 + self.Fj
        self.W_fo = (signal_m_function(value['Fj']) * signal_m_function(value['Oj']) * (signal_m_function(value['Oj']) - self.W_fo)) * self.dt/50 + self.W_fo
        # self.Oj = self.build_Invariant_object_category(self.Vjiq)
        self.W_ov = (signal_m_function(value['Oj']) * F.relu(value['Vjiq']) * (F.relu(value['Vjiq']) - self.W_ov)) * self.dt/50 + self.W_ov

    def train(self, input):
        self.part1(input)
        self.Eye_movements_map()

        input_B = self.Vector_Bq(self.B)
        _,_,W,V = self.fuzzy_ART.train_(input_B, self.Vjiq)
        self.part2_up(V)
        self.part2_down()

    def habituative_transmitter(self, inter_value):
        self.Y_ij = self.Y_ij + self.dt * (self.K_e * (2 - self.Y_ij - 3 * 10 ** 6 * self.Y_ij * (F.relu(self.Cij) + 625 * inter_value)))

    def sign_h(self):
        max_value = torch.max(self.Eij)
        h_Eij = torch.zeros_like(self.Eij)
        if max_value>0.58:
            max_place = (self.Eij==torch.max(self.Eij)).nonzero()[0]
            h_Eij[max_place[0]][max_place[1]][max_place[2]][max_place[3]] = 1
        return h_Eij

        # if self.Eij == max(self.Eij) and max(self.Eij) > 0.58:
        # h_Eij = 1
        # return h_Eij


    def Eye_movements_map(self):
        dt, E_ij, C_ij, Y_ij, M = self.dt, self.Eij, self.Cij, self.Y_ij
        K_kernel = gaussianKernel(3, 5, 1/(50*np.pi))
        J_kernel = gaussianKernel(3, 1, 1/(2*np.pi))
        #A45
        inter_value_J = gaussianconv_2d(E_ij**2, J_kernel, 3)
        #A46
        inter_value_K = gaussianconv_2d(E_ij**2, K_kernel, 3)
        #A47
        self.habituative_transmitter(inter_value_J)

        self.W_ve = signal_m_function(self.Vjq) * self.sign_h() * (self.Eij - self.W_ve) * 500 * self.dt + self.W_ve

        self.Eij = (self.Y_ij * (1 - self.Eij) * (F.relu(C_ij) + 625 * inter_value_J + torch.sum(signal_m_function(self.Vjq)*self.W_ve)) - 0.01 * self.Eij * torch.sum(F.relu(self.Cij + inter_value_K)) - self.Eij ) * self.dt + self.Eij

        max_value = torch.max(self.Eij)
        EIJ = torch.zeros_like(self.Eij)
        if max_value>0.58:
            max_place = (self.Eij==torch.max(self.Eij)).nonzero()[0]
            EIJ[max_place[0]][max_place[1]][max_place[2]][max_place[3]] = max_value
        return EIJ

    #A60
    def normalized_V_(self, V):
        mask = (V == V.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        V = torch.mul(mask, V)
        return V

    def normalized_O_(self, Oj):
        Oj_on = F.relu(Oj)
        mask = (Oj_on == Oj_on.max(dim=0, keepdim=True)[0]).to(dtype=torch.int32)
        Oj = torch.mul(mask, Oj_on)
        return Oj

    def normalized_D_(self, Dj):
        mask = (Dj == Dj.max(dim=0, keepdim=True)[0]).to(dtype=torch.int32)
        Dj = torch.mul(mask, Dj)
        return Dj

    def normalized_F_(self,Fj):
        mask = (Fj == Fj.max(dim=0, keepdim=True)[0]).to(dtype=torch.int32)
        Fj = torch.mul(mask, Fj)
        return Fj

    def normalized_N_(self, Nj):
        mask = (Nj == Nj.max(dim=0, keepdim=True)[0]).to(dtype=torch.int32)
        Nj = torch.mul(mask, Nj)
        return Nj

    def normalized_V(self, V):
        original = V
        result = torch.zeros_like(original)
        key = torch.arange(0,original.shape[1],1)!=torch.arange(0,original.shape[1],1)

        for i in range(original.shape[0]):
            max_index = torch.argmax(original[i])
            min_index = torch.argmin(original[i])
            if sum(original[i][max_index] - original[i] >= 2) == original.shape[1] -1:
                result[i][max_index] = original[i][max_index]
                # part1 = [max_index, original[i][max_index]]
                key = torch.arange(0,original.shape[1],1) == max_index
            part2 = original[i][~key]
            cut = torch.abs(original[i][~key].unsqueeze(dim=1)-part2)<=0.03
            cut = torch.sum(cut,dim=1) >= original.shape[1]
            if sum(cut) > 0 and  min(part2[~cut]) - max(part2[cut]) > -0.03:
                index = index[~key][cut]
                value = original[i][index] **2 / (torch.sum(part2)-original[i][index])
                result[i][index] = value
        Vjq = result        
        # V[0][1] = 1.104 #example
        # Vjq_value, _ = V.sort(descending=True)
        # min_value = Vjq_value[:,-1:]
        # max_value = Vjq_value[:,:1]
        # Vjq_value = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), Vjq_value), dim=1)  #增加一列index列
        # key = torch.sum(((max_value - (Vjq_value[:,2:] + 0.03)) > 0), dim=1) == (Vjq_value.shape[1]-2)
        # # key = key.expand(100,25).transpose(0,1)
        # #A60/1
        # part1 = Vjq_value[key]  
        # #A60/2
        # key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.03 #实验数据设置0.5546
        # part2 = Vjq_value[~key][key2]
        # part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
        
        # part3 = Vjq_value[~key][~key2]
        # part3[:,1:] = part3[:,1:] * 0
        # Vjq = torch.cat((part1,part2,part3),dim=0)
        # # torch.masked_select(Vjq_value,key.expand(100,25).transpose(0,1))
        # Vjq = sorted(Vjq, key = lambda x:x[0])
        # Vjq = torch.tensor([item.cpu().detach().numpy() for item in Vjq])[:,1:]
        return Vjq     

    def normalized_N(self):
        self.Nj, _ = self.Nj.sort(descending=True)
        min_value = self.Nj[:,-1:]
        max_value = self.Nj[:,:1]
        self.Nj = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), self.Nj), dim=1)  #增加一列index列
        key = torch.sum(((max_value - (self.Nj[:,2:] + 0.1)) > 0), dim=1) == (self.Nj.shape[1]-2)
        # key = key.expand(100,25).transpose(0,1)
        #A60/1
        part1 = self.Nj[key]  
        #A60/2
        key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.1 #实验数据设置0.5546
        part2 = self.Nj[~key][key2]
        part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
        
        part3 = self.Nj[~key][~key2]
        part3[:,1:] = part3[:,1:] * 0
        self.Nj = torch.cat((part1,part2,part3),dim=0)
        # torch.masked_select(Vjq_value,key.expand(100,25).transpose(0,1))
        self.Nj = sorted(self.Nj, key = lambda x:x[0])
        self.Nj = torch.tensor([item.cpu().detach().numpy() for item in self.Nj])[:,1:]
        return self.Nj

    def normalized_D(self):
        self.Dj, _ = self.Dj.sort(descending=True)
        min_value = self.Dj[:,-1:]
        max_value = self.Dj[:,:1]
        self.Dj = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), self.Dj), dim=1)  #增加一列index列
        key = torch.sum(((max_value - (self.Dj[:,2:] + 0.1)) > 0), dim=1) == (self.Dj.shape[1]-2)
        # key = key.expand(100,25).transpose(0,1)
        #A60/1
        part1 = self.Dj[key]  
        #A60/2
        key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.1 #实验数据设置0.5546
        part2 = self.Dj[~key][key2]
        part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
        
        part3 = self.Dj[~key][~key2]
        part3[:,1:] = part3[:,1:] * 0
        self.Dj = torch.cat((part1,part2,part3),dim=0)
        # torch.masked_select(Vjq_value,key.expand(100,25).transpose(0,1))
        self.Dj = sorted(self.Dj, key = lambda x:x[0])
        self.Dj = torch.tensor([item.cpu().detach().numpy() for item in self.Dj])[:,1:]
        return self.Dj

    def normalized_F(self):
        self.Fj, _ = self.Fj.sort(descending=True)
        min_value = self.Fj[:,-1:]
        max_value = self.Fj[:,:1]
        self.Fj = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), self.Fj), dim=1)  #增加一列index列
        key = torch.sum(((max_value - (self.Fj[:,2:] + 0.5)) > 0), dim=1) == (self.Fj.shape[1]-2)
        # key = key.expand(100,25).transpose(0,1)
        #A60/1
        part1 = self.Fj[key]  
        #A60/2
        key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.5 #实验数据设置0.5546
        part2 = self.Fj[~key][key2]
        part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
        
        part3 = self.Fj[~key][~key2]
        part3[:,1:] = part3[:,1:] * 0
        self.Fj = torch.cat((part1,part2,part3),dim=0)
        # torch.masked_select(Vjq_value,key.expand(100,25).transpose(0,1))
        self.Fj = sorted(self.Fj, key = lambda x:x[0])
        self.Fj = torch.tensor([item.cpu().detach().numpy() for item in self.Fj])[:,1:]
        return self.Fj

    def normalized_O(self):
        Oj_on = F.relu(self.Oj)
        Oj_on, _ = Oj_on.sort(descending=True)
        min_value = Oj_on[0]
        max_value = Oj_on[-1]

        if max_value - Oj_on[1] > 0.3:
            return Oj_on

        Oj_on = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), Oj_on), dim=1)
        key = torch.sum(((max_value - (Oj_on[:,2:] + 0.3)) > 0), dim=1) == (Oj_on.shape[1]-2)
        part1 = Oj_on[key] 

        key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.3
        part2 = Oj_on[~key][key2]
        part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
        
        part3 = Oj_on[~key][~key2]
        part3[:,1:] = part3[:,1:] * 0
        Oj_on = torch.cat((part1,part2,part3),dim=0)
        Oj_on = sorted(Oj_on, key = lambda x:x[0])
        Oj_on = torch.tensor([item.cpu().detach().numpy() for item in Oj_on])[:,1:]
        return Oj_on

    def build_Invariant_object_category(self, Vjiq):
        self.Oj = self.Oj + self.dt * (-self.Oj + (1 + 2 * self.Fj * self.W_fo) * (0.5 * torch.sum(signal_m_function(half_rectified_relu(Vjiq))* self.W_vo)  + self.G) - self.Rwhere)
        self.Oj_normalized(self.Oj)


    #A61    
    def build_View_category_integrator(self, Vjq):
        self.Vjiq = (-0.01 * self.Vjiq + self.t * (1 + torch.sum(signal_m_function(self.Oj) * self.W_ov)) * (half_rectified(Vjq) + self.G) - self.Rwhere) * self.dt + self.Vjiq

    def get_G(self, key):
        if key == "down":
            return 0.1
        else:
            return 0        
    
    def Vector_Bq(self, input):
        input = torch.cat(torch.split(input[0][0], 100, 0),dim=1)
        output = torch.split(input, 100, 1)
        output = torch.cat([fm.unsqueeze(0).reshape(-1,10000) for fm in output], dim=0)
        return output

    #A6 surface contours
    def Surface_contours(self):
        Jpq_on = 0.8 * self.Object_surface_ons[0] + 0.1 * self.Object_surface_ons[1] + 0.1 * self.Object_surface_ons[2]
        Jpq_off = 0.8 * self.Object_surface_offs[0] + 0.1 * self.Object_surface_offs[1] + 0.1 * self.Object_surface_offs[2]
        K_on_kernel = gaussianKernel(ksize=5, sigma=1, c=1/(2*np.pi))
        K_off_kernel = gaussianKernel(ksize=5, sigma=3, c=1/(2*np.pi*9))
        J = Jpq_on - Jpq_off
        Kernel_add = K_on_kernel + K_off_kernel
        Kernel_cut = K_on_kernel - K_off_kernel
        Kernel_cut1 = K_off_kernel - K_on_kernel
        # plt.subplot(4, 3, 1)   #plt.subplot(nrows, ncols, index)
        # plt.imshow(Kernel_add)
        molecular = gaussianconv_2d(J, Kernel_cut, 5)
        denominator = 40 + gaussianconv_2d(J, Kernel_add, 5)
        molecular1 = gaussianconv_2d(J, Kernel_cut1, 5)
        output_C = half_rectified_relu(molecular / denominator) + half_rectified(molecular1 / denominator)
        return output_C

    #A22
    def Surface_filling_in_(self, Boundary, S_filling_ins, X_input):
        output_signal_sf = self.Sijf
        S_filling_ins = self.Object_surface_ons
        Pboundary_gated_diffusion = boundary_gated_diffusion(Boundary)
        s_with_p = S_reduce_with_P(S_filling_ins, Pboundary_gated_diffusion)

        output = []
        for i  in range(len(S_filling_ins)):
            S_filling_in = S_filling_ins[i]
            #A22
            output.append((S_filling_in + (-80) * S_filling_in + s_with_p[i] + 100 * X_input[i] * (1 + output_signal_sf) - S_filling_in * self.Rwhere) * self.dt)
        return output

    #A18
    def Boundaries(self, Z_complex_cells):
        # Z_complex_cells = argument['Z']
        C_Surface_contours = self.Cij
        M = self.M
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
 


def main():
    global argument
    argument = {}
    argument["dt"] = 0.05

    img = data_process("/Users/leilei/Desktop/ARTSCAN/image.jpg")
    #A.1. Retina and LGN cells
    imgsc_on, imgsc_off = GaussianBlur(img, [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7])
    #A.2. V1 polarity-sensitive oriented simple cells
    model = fuzzy_ART(X_size=100 * 100, c_max=100, rho=0.85, alpha=0.00001, beta=1)
    #A13
    Y_ons, Y_offs = Gabor_conv(imgsc_on, imgsc_off, sigmav= [3, 4.5, 6], sigmah = [1, 1.5, 2], Lambda = [3, 5, 7], angles = [0, 45, 90, 135], K_size = [(19,19), (29,29), (39,39)])
    #A16
    Z = complex_cells(Y_ons, Y_offs)
    argument["Z"] = Z 
    argument["imgsc_on"] = imgsc_on
    argument["imgsc_off"] = imgsc_off
    model1 = arcscan(size=(500,500))
    for t in range(1,40):
        model1.train(argument)

        # #A23
        # Object_surface_ons = Surface_filling_in_(argument, argument, argument["Object_surface_ons"], imgsc_on)
        # Object_surface_offs = Surface_filling_in_(argument, argument, argument["Object_surface_offs"], imgsc_off)
        # argument["Object_surface_ons"] = Object_surface_ons
        # argument["Object_surface_offs"] = Object_surface_offs
        # # Object_surface_ons = Surface_filling_in(dt, Pboundary_gated_diffusion, Object_surface_ons, output_signal_sf,  imgsc_on)
        # #A23
        # # Object_surface_offs = Surface_filling_in(dt, Pboundary_gated_diffusion, Object_surface_offs, output_signal_sf, imgsc_off)
        # #A26
        # argument["S_ij"] = 0.05 * (Object_surface_ons[0] + Object_surface_offs[0]) + 0.1 * (Object_surface_ons[1] + Object_surface_offs[1]) + 0.85 * (Object_surface_ons[2] + Object_surface_offs[2])
        # argument = Surface_contours(argument)
        # #A48
        # W = signal_m_function(Vjq)
        # #A43
        EIJ, Eij, Y_ij, max_place = Eye_movements_map(argument)
        # #A34  max_place, Amn
        # #临时定义
        # max_place = [0,0,50,150]
        # SFmn, AImn = Gain_field(argument, max_place, Amn)




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
        output.append(((-80) * S_filling_in + Boundary_diffusionP[i] * S_filling_in_conv + 100 * X_input[i] * (1 + output_signal_sf) - S_filling_in * Rwhere) * dt + S_filling_in)
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

if __name__ == "__main__":
    main()

