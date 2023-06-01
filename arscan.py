import torch
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import math
from utils.conv2d import gaussianconv_2d, Gabor_conv, GaussianBlur
from utils.kernel import Gabor_filter, gaussianKernel, getGaussianKernel
from utils.utils import half_rectified, type_input, data_process, signal_m_function, half_rectified_relu, signal_g_function, signal_f_function, signal_k_function
from fuzzy_ART import fuzzy_ART
from utils.utils import boundary_gated_diffusion, S_reduce_with_P, complex_cells
import os
class arcscan():
    def __init__(self, size=(500, 500), category=12, dt = 0.001, t=0.6, Rwhere=0, key="up"):
        self.device="cuda"
        self.dt = dt
        self.size = size
        self.batch = 25
        self.category = category + 1

        self.B = torch.zeros(size).to(self.device)
        self.Object_surface_ons = [torch.zeros(size).to(self.device), torch.zeros(size).to(self.device), torch.zeros(size).to(self.device)]
        self.Object_surface_offs = [torch.zeros(size).to(self.device), torch.zeros(size).to(self.device), torch.zeros(size).to(self.device)]
        self.S_ij = torch.zeros(size).to(self.device)
        self.Cij = torch.zeros(1, 1, size[0], size[1]).to(self.device)

        self.Eij = torch.zeros(size).to(self.device)
        self.Y_ijE = 2 * torch.ones(size).to(self.device)

        self.Y_ijA = 2 * torch.ones(size).to(self.device)
        self.Amn = torch.zeros(1, 1, size[0], size[1]).to(self.device)
        self.Sijf = torch.zeros(size).to(self.device)
        # self.W_ve = torch.ones((self.category,size[0],size[1]))
        self.W_ve = 0.01 * torch.ones((25, self.category, size[0], size[1])).to(self.device)
        
        self.fuzzy_ART = fuzzy_ART(X_size=100 * 100, c_max=self.category, rho=0.85, alpha=0.00001, beta=1)
        self.W = torch.ones( (self.category, 20000) ).to(self.device)
        self.t = t
        self.Vjq = torch.zeros( (self.batch, self.category) ).to(self.device)

        self.t_view = [{"Vjiq": torch.zeros( (self.batch, self.category) ).to(self.device),
                        "Oj": torch.zeros( (self.category) ).to(self.device), 
                        "Fj": torch.zeros( (self.category) ).to(self.device), 
                        "Dj": torch.zeros( (self.category) ).to(self.device), 
                        "Nj": torch.zeros( (self.category) ).to(self.device)},
                       {"Vjiq": torch.zeros( (self.batch, self.category) ).to(self.device),
                        "Oj": torch.zeros( (self.category) ).to(self.device), 
                        "Fj": torch.zeros( (self.category) ).to(self.device), 
                        "Dj": torch.zeros( (self.category) ).to(self.device), 
                        "Nj": torch.zeros( (self.category) ).to(self.device)}]

        #up
        self.W_vo = 0.01 * torch.ones( (self.batch, self.category, self.category) ).to(self.device)
        self.W_ov = 0.01 * torch.ones( (self.batch, self.category, self.category) ).to(self.device)

        self.W_od = 0.01 * torch.ones( (self.category, self.category) ).to(self.device)
        self.W_df = 0.01 * torch.ones( (self.category, self.category) ).to(self.device)

        self.W_fn = 0.01 * torch.ones( (self.category, self.category) ).to(self.device)
        self.W_nf = 0.01 * torch.ones( (self.category, self.category) ).to(self.device)

        self.W_of = 0.01 * torch.ones( (self.category) ).to(self.device)    
        self.W_fo = 0.01 * torch.ones( (self.category) ).to(self.device)

        self.Rwhere = Rwhere
        self.yR = 0

        self.key = key
        self.G = self.get_G(self.key)
        self.Uj = 0
        self.Tj = 0
        self.Pj = 0

        self.eye_move = False
        self.max_place = (249, 249)
        self.max_place_pre = (249, 249)
        self.max_place_pre_in_eye = (499, 499)
        self.max_place_in_eye = (499, 499)
        self.K_e = 10**-7

    def update(self, value):
        self.t_view[0] = self.t_view[1]
        self.t_view[1] = value

    def part1(self, input):
        input = input.to(self.device)
        #A.1. Retina and LGN cells
        imgsc_on, imgsc_off = GaussianBlur(input, [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7])
        #A.2. V1 polarity-sensitive oriented simple cells
        # model = fuzzy_ART(X_size=100 * 100, c_max=100, rho=0.85, alpha=0.00001, beta=1)
        #A13
        Y_ons, Y_offs = Gabor_conv(imgsc_on, imgsc_off, sigmav= [3, 4.5, 6], sigmah = [1, 1.5, 2], Lambda = [3, 5, 7], angles = [0, 45, 90, 135], K_size = [(19,19), (29,29), (39,39)])
        #A16
        Z = complex_cells(Y_ons, Y_offs)
        # type_input(Z, "Z", 1)
        # argument["Z"] = Z 
        # argument["imgsc_on"] = imgsc_on
        # argument["imgsc_off"] = imgsc_off

        # Z = input["Z"], 
        # imgsc_on =  input["imgsc_on"]
        # imgsc_off = input["imgsc_off"]
        # for i in range(20):
        self.B = self.Boundaries(Z)
        # type_input(self.B, "B", 1)
        self.Object_surface_ons = self.Surface_filling_in_(self.B, self.Object_surface_ons, imgsc_on)
        # type_input(self.Object_surface_ons, "Object_surface_ons", 1)
        self.Object_surface_offs = self.Surface_filling_in_(self.B, self.Object_surface_offs, imgsc_off)
        # type_input(self.Object_surface_offs, "Object_surface_offs", 1)
        self.Cij = self.Surface_contours()
        # type_input(self.Cij, "Cij", 1)
        self.S_ij = 0.05 * (self.Object_surface_ons[0] + self.Object_surface_offs[0]) + 0.1 * (self.Object_surface_ons[1] + self.Object_surface_offs[1]) + 0.85 * (self.Object_surface_ons[2] + self.Object_surface_offs[2])
        # type_input(self.S_ij, "S_ij", 1)
        self.Eye_movements_map()
        self.Gain_field()
        self.attention_shroud()
        self.Reset()    
        self.Y_R()    

    def Y_R(self):
        self.yR = 6*(2-self.yR - 2*self.yR *self.Rwhere) * self.dt + self.yR
    # return self.B
    #view category integrators
    def part2_up(self, input):
        self.Vjq = self.Normalized2d(input)
        value = self.t_view[0]
        #Vjiq
        Vjiq = (-0.01 * value["Vjiq"] + self.t * ((1 + torch.sum(self.W_ov * signal_m_function(value["Oj"]))) * (F.relu(self.Vjq) + self.G)) - self.Rwhere) * self.dt + value["Vjiq"]
        
        #A63
        Oj = value['Oj'] + self.dt * (-value['Oj'] + (1 + 2 * value['Fj'] * self.W_fo) * (0.5 * self.neuron3d(signal_m_function(F.relu(value['Vjiq'])), self.W_vo)  + self.G) - self.Rwhere)
        Oj = self.Normalized1d(F.relu(Oj))
        
        #A66
        Dj = -value['Dj'] + self.Uj + self.Tj + 0.1 * self.neuron1d(signal_m_function(value['Oj']), self.W_od)   #待修改
        Dj = self.Normalized1d(Dj)

        Fj = (-value['Fj'] + (0.5 * signal_m_function(value['Oj']) * self.W_of + self.G) * (1 + self.neuron1d(value['Dj'], self.W_df) + self.neuron1d(signal_m_function(value['Nj']), self.W_nf))) * self.dt * 20 + value['Fj']
        Fj = self.Normalized1d(Fj)

        Nj = (- value['Nj'] + self.neuron1d(signal_m_function(value['Fj']), self.W_fn) + self.Pj) * 20 * self.dt
        Nj = self.Normalized1d(Nj)
        value_dic = {"Vjiq":Vjiq, "Oj":Oj, "Dj":Dj, "Fj":Fj, "Nj":Nj}
        self.update(value_dic)
        
    def neuron2d(self, input, W):
        sum_ne = 0
        for i in range(input.shape[0]):
            sum_ne += (input[i]*W.unsqueeze(dim=2)).sum()
        return sum_ne

    def neuron1d(self, input, W):
        sum_ne = (input*W.unsqueeze(dim=2)).sum()
        return sum_ne

    def neuron3d(self, input, W):
        sum_ne = 0
        for i in range(input.shape[0]):
            sum_ne += (input[i]*W[i].unsqueeze(dim=2)).sum()
        return sum_ne

        # self.W_ov = (signal_m_function(self.Oj) * F.relu(self.Vjiq) * (F.relu(self.Vjiq) - self.W_ov)) * self.dt + self.W_ov

    def part2_down(self):
        value = self.t_view[1]
        #A75
        self.W_ov = (signal_m_function(value['Oj']) * F.relu(value['Vjiq']).unsqueeze(dim=2) * self.W_reduce(F.relu(value['Vjiq']), self.W_ov)) * self.dt/50 + self.W_ov
        #A76
        self.W_of = (signal_m_function(value['Oj'])* self.W_reduce(signal_m_function(value['Fj']), self.W_of)) * self.dt / 50 + self.W_of
        #A77
        self.W_fo = (signal_m_function(value['Fj']) * signal_m_function(value['Oj']) * self.W_reduce(signal_m_function(value['Oj']), self.W_fo)) * self.dt/50 + self.W_fo
        #A78
        self.W_od = signal_m_function(value['Oj']) * signal_m_function(value['Dj']) * self.W_reduce(signal_m_function(value['Dj']), self.W_od) * self.dt / 50 + self.W_od
        #A79
        self.W_df = signal_m_function(value['Dj']) * signal_m_function(value['Fj']) * self.W_reduce(signal_m_function(value['Fj']), self.W_df) * self.dt/50+ self.W_df
        #80
        self.W_fn = signal_m_function(value['Fj']) * self.W_reduce(signal_m_function(value['Nj']), self.W_fn) * self.dt /50 + self.W_fn
        
        self.W_nf = signal_m_function(value['Nj']) * signal_m_function(value['Fj']) * self.W_reduce(signal_m_function(value['Fj']), self.W_nf) * self.dt /50 + self.W_nf
        # self.Fj = (-self.Fj + (0.5 * signal_m_function(self.Oj) * self.W_of + self.G) * (1 + torch.sum(self.Dj * self.W_df + torch.sum(signal_m_function(self.Ni) * self.W_nf)))) * self.dt * 20 + self.Fj
        # self.Oj = self.build_Invariant_object_category(self.Vjiq)
    
    def W_reduce(self, input, W):
        output = []
        for i in range(input.shape[0]):
            output.append((input[i]-W[i]).unsqueeze(dim=0))
        return torch.cat(output,dim=0)

    def train(self, input):
        self.part1(input)
        input_B = self.Vector_Bq(self.B)
        _,_,self.W,V = self.fuzzy_ART.train_(input_B, self.t_view[0]["Vjiq"])
        self.part2_up(V)
        self.part2_down()

    # B.4. Object category reset by transient parietal bursts
    #A50
    def Reset(self, e=0.07):
        self.Rwhere = 100 * F.relu(100/(100 + signal_k_function(self.Amn).sum()) - e)

    # Gain field
    def Gain_field(self):
        shifted_S = self.shifted_map(self.S_ij, 1)
        shifted_A = self.shifted_map(self.Amn, 0)
        #A34--AImn
        self.Amni = shifted_S + self.Amn
        #A36--SFmn
        self.Sijf = shifted_A + self.S_ij
        return self.Amni, self.Sijf

    #A37
    def attention_shroud(self):
        kernel_C = gaussianKernel(19, 4, 0.1)
        kernel_E = gaussianKernel(19, 200, 1/(10**5 * 2))
        gA = signal_g_function(self.Amni)
        fA = signal_f_function(self.Amn)
        self.habituative_transmitter_A()
        input_A = gA * (1 + 0.2 * gaussianconv_2d(fA, kernel_C, 19)) * (1 - self.Amn) *self.Y_ijA - 0.1*self.Amn        
        #***
        second = -self.Amn * gaussianconv_2d(gA + fA, kernel_E.float(), 19) + 10 * self.Rwhere * self.yR
        self.Amn = (input_A + second) * 10 * self.dt + self.Amn

    def Eye_movements_map(self):
        K_kernel = gaussianKernel(35, 5, 1/(2 * np.pi*25))
        J_kernel = gaussianKernel(5, 1, 1/(2*np.pi))
        #A45
        inter_value_J = gaussianconv_2d((self.Eij**2).unsqueeze(dim=0).unsqueeze(dim=1).float(), J_kernel, 5)
        #A46
        inter_value_K = gaussianconv_2d((self.Eij**2).unsqueeze(dim=0).unsqueeze(dim=1).float(), K_kernel, 35)
        #A47
        self.habituative_transmitter_E(inter_value_J)
        Vjq = signal_m_function(self.Vjq)

        h_Eij = self.sign_h()
        if  h_Eij == 1:
            # for i in range(self.Vjq.shape[0]):
            #     sumve += (Vjq[i] * (self.Eij - self.W_ve.transpose(0,2).transpose(1,3)).unsqueeze(dim=2))
            #     self.W_ve = sumve * h_Eij * 500 * self.dt + self.W_ve
            self.W_ve = (Vjq * (self.Eij - self.W_ve.transpose(0,2).transpose(1,3))).transpose(0,2).transpose(1,3)
        # sumvw = 0
        sumvw = torch.sum(torch.sum(Vjq * self.W_ve.transpose(0,2).transpose(1,3), dim=3),dim=2)
        # for q in range(self.Vjq.shape[0]):
        #     # for k in range(self.Vjq.shape[1]):
        #         sumvw += torch.sum(Vjq[q]*self.W_ve[q].transpose(0,2),dim=2)

        Eij = F.relu(self.Cij)+625*inter_value_J + sumvw
        Eij = self.Y_ijE*(1-self.Eij.unsqueeze(dim=0).unsqueeze(dim=0))*Eij
        Eij = Eij - 0.01 * self.Eij * torch.sum(F.relu(self.Cij) + inter_value_K)
        Eij = Eij-self.Eij
        self.Eij = (Eij * self.dt + self.Eij).squeeze(dim=0).squeeze(dim=0)

        max_value = torch.max(self.Eij)
        if max_value>0.58:  #0.58
            self.eye_move = True
            self.max_place_pre = self.max_place
            self.max_place = (self.Eij==torch.max(self.Eij)).nonzero()[0]
            self.max_place = (int(self.max_place[0]), int(self.max_place[1]))
            max_0 = self.max_place[0] - self.max_place_pre[0] + self.max_place_in_eye[0]
            max_1 = self.max_place[1] - self.max_place_pre[1] + self.max_place_in_eye[1]
            self.max_place_in_eye = (self.max_restriction(max_0),self.max_restriction(max_1))
        else:
            self.eye_move = True
            if self.eye_move:
                print(self.max_place_in_eye, self.max_place)
            # self.Eij[self.max_place[0]][self.max_place[1]] = max_value
    def max_restriction(self, max_0):
            if max_0 > 749 :
                output = 749
            elif max_0 < 249:
                output = 249
            else:
                output = max_0
            return output
    
    def shifted_map(self, input, key):
        B, C, H, W = input.shape
        #定义扩张后的眼球map
        c=torch.zeros((B, C, H*2, W*2)).cuda()
        #赋值0.5后嵌入原S的视图
        c[:] = 0.5
        #中间位置y,x平移(k,l)个单位A32中的S(m-k,n-l)
        # if self.max_place is None:
        if key:
            c[:, :, H//2-1:int(1.5*H)-1, W//2:int(1.5*W)] = input
            output = c[:, :, self.max_place[0]:500 + self.max_place[0], self.max_place[1]: 500 + self.max_place[1]]
        else:
            c[:, :, self.max_place[0]:500 + self.max_place[0], self.max_place[1]: 500 + self.max_place[1]] = input
            output = c[:, :, H//2-1:int(1.5*H)-1, W//2:int(1.5*W)]
        return output
        # else:
        #     return input
        

    #A42
    def habituative_transmitter_A(self):
        kernel_C = gaussianKernel(29, 4, 0.1)
        gA = signal_g_function(self.Amni)
        fA = signal_f_function(self.Amn)
        fA = gaussianconv_2d(fA, kernel_C, 29)
        input_A = 3 * 10**6 * gA * (1 + 0.2 * fA) * self.Y_ijA
        self.Y_ijA = 10**-8 * (2 - self.Y_ijA - input_A) * self.dt + self.Y_ijA


    def habituative_transmitter_E(self, inter_value):
        self.Y_ijE = self.Y_ijE + self.dt * (self.K_e * (2 - self.Y_ijE - 3 * 10 ** 6 * self.Y_ijE * (F.relu(self.Cij) + 625 * inter_value)))

    def sign_h(self):
        max_value = torch.max(self.Eij)
        # h_Eij = torch.zeros_like(self.Eij)
        if max_value>0.58:
            # max_place = (self.Eij==torch.max(self.Eij)).nonzero()[0]
            # h_Eij[max_place[0]][max_place[1]][max_place[2]][max_place[3]] = 1
            h_Eij = 1
        else:
            h_Eij = 0
        return h_Eij
        

        # if self.Eij == max(self.Eij) and max(self.Eij) > 0.58:
        # h_Eij = 1
        # return h_Eij

    #A60
    def normalized1d(self, input):
        mask = (input == input.max(dim=0, keepdim=True)[0]).to(dtype=torch.int32)
        input = torch.mul(mask, input)
        return input
       
    def normalized2d(self, V):
        mask = (V == V.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        V = torch.mul(mask, V)
        return V

    def Normalized2d(self, input):
        output = []
        for i in range(input.shape[0]):
            output.append(self.Normalized1d(input[i]))
        return torch.cat(output).reshape_as(input)


    def Normalized1d(self, input):
        result = torch.zeros_like(input)

        out, ind = input.sort(descending=True)
        if (out[0] - out[1]) >0.1:
            result[ind[0]] = out[0]
            out = out[1:]
            ind = ind[1:]

        key = torch.abs(out[:-1] - out[1:])<=0.1
        if torch.sum(key) == len(key):
            key_0 = len(key)
        elif torch.sum(key)==0:
            return result
        else:
            key_0 = list(np.array(key.cpu())).index(0)

        out2 = out[: key_0+1]
        ind1 = ind[: key_0+1]

        part2 = (out2/(torch.sum(out2))) * out2

        result[ind1] = part2
        
        return result

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
        input = torch.cat(torch.split(input[2][0][0], 100, 0),dim=1)
        output = torch.split(input, 100, 1)
        output = torch.cat([fm.unsqueeze(0).reshape(-1,10000) for fm in output], dim=0)
        return output

    #A6 surface contours
    #A27
    def Surface_contours(self):
        Jpq_on = 0.8 * self.Object_surface_ons[0] + 0.1 * self.Object_surface_ons[1] + 0.1 * self.Object_surface_ons[2]
        Jpq_off = 0.8 * self.Object_surface_offs[0] + 0.1 * self.Object_surface_offs[1] + 0.1 * self.Object_surface_offs[2]
        K_on_kernel = gaussianKernel(ksize=17, sigma=1, c=1/(2 * np.pi))
        K_off_kernel = gaussianKernel(ksize=17, sigma=3, c = 1/(2*np.pi*9))
        J = Jpq_on - Jpq_off
        Kernel_add = K_on_kernel + K_off_kernel
        Kernel_cut = K_on_kernel - K_off_kernel
        Kernel_cut1 = K_off_kernel - K_on_kernel
        # plt.subplot(4, 3, 1)   #plt.subplot(nrows, ncols, index)
        # plt.imshow(Kernel_add)
        molecular = gaussianconv_2d(J, Kernel_cut, 17)
        denominator = 40 + gaussianconv_2d(J, Kernel_add, 17)
        molecular1 = gaussianconv_2d(J, Kernel_cut1, 17)
        output_C = F.relu(molecular / denominator) + F.relu(molecular1 / denominator)
        return output_C

    #A22
    def Surface_filling_in_(self, Boundary, S_filling_ins, X_input):
        S_filling_ins = self.Object_surface_ons
        Pboundary_gated_diffusion = boundary_gated_diffusion(Boundary)
        s_with_p = S_reduce_with_P(S_filling_ins, Pboundary_gated_diffusion)

        output = []
        for i  in range(len(S_filling_ins)):
            S_filling_in = S_filling_ins[i]
            #A22
            output.append((S_filling_in + (-80) * S_filling_in + s_with_p[i] + 100 * X_input[i] * (1 + self.Sijf) - S_filling_in * self.Rwhere) * self.dt)
        return output


    #A18
    def Boundaries(self, Z_complex_cells):
        # Z_complex_cells = argument['Z']


        Vjq = signal_m_function(self.Vjq)

        sumvb = 0
        for i in range(self.Vjq.shape[0]):
            sumvb += torch.sum(Vjq[i]*self.W[:,:10000].transpose(1,0))

        output = []
        for i in range(len(Z_complex_cells)):
            #A20
            Gs_F = gaussianKernel(ksize=35, sigma=5, c=1/(2*np.pi * 25))

            C_S2d = gaussianconv_2d(self.Cij, Gs_F, 5) 
            if i == 2:
                B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d + sumvb) + 0.4 * self.Cij.sum()
                B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d + sumvb) - 0.4 * self.Cij.sum()
                B = F.relu(B_molecular/B_denominator)
                output.append(B)
            else:
                B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d) + 0.4 * self.Cij.sum()
                B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d) - 0.4 * self.Cij.sum()
                B = F.relu(B_molecular/B_denominator)
                output.append(B)
        return output
 


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global argument
    argument = {}

    imgs = data_process("./example.jpg", (1000, 1000))
    
    model1 = arcscan(size=(500,500))
    eye_path = cv2.imread("./example.jpg")
    
    for t in range(300):
        img = imgs[:, :, model1.max_place_in_eye[0]-250:model1.max_place_in_eye[0]+250, model1.max_place_in_eye[1]-250:model1.max_place_in_eye[1]+250]
        model1.train(img)
        print((int(model1.max_place[1]), int(model1.max_place[0])))
        print((model1.max_place_pre[1],model1.max_place_pre[0]), (model1.max_place[1],model1.max_place[0]))
        img1 = cv2.circle(eye_path,(int(model1.max_place[1]), int(model1.max_place[0])),5,(0,0,255),-1)
        img1 = cv2.line(img1, (model1.max_place_pre[1],model1.max_place_pre[0]), (model1.max_place[1],model1.max_place[0]), (0, 255, 0), 2)
        cv2.imwrite("./output_path.jpg", img1)
           
if __name__ == "__main__":
    main()

