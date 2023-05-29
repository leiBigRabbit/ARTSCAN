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
from utils.utils import half_rectified, type_input, data_process, signal_m_function, half_rectified_relu, signal_g_function, signal_f_function, signal_k_function
from eye_movements import Eye_movements_map
from Spatial_attentional import attention_shroud
from gain import Gain_field
from what_stream import Vector_Bq, activity_V, what_stream
from fuzzy_ART import fuzzy_ART
from View_category_integrator import View_category_integrator
from Invariant_object_category import object_category_neuron


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
    # type_input(outputs, "sp", 1)
    return outputs

#A16
def complex_cells(Y_ons, Y_offs):
    #A15
    Y_add = np.array(Y_ons).sum(axis=0)
    Y_cuts = np.array(Y_offs).sum(axis=0)
    input = np.array([Y_add, Y_cuts]).sum(axis=0)
    # type_input(input, "output_on1", 1)
    #A17
    L_gskernel = gaussianKernel(5, 1, 1/(2*np.pi))
    output = []
    for Z in input:
        complex = F.relu(Z ** 2 / (0.01 + gaussianconv_2d(Z**2, L_gskernel, (5,5))))
        output.append(complex)
    return output

class arcscan():
    def __init__(self, size=(500, 500), dt = 0.05, t=0.6, Rwhere=0, key="up"):
        self.device="cpu"
        self.dt = dt
        self.size = size
        self.batch = 25
        self.size_ = (self.batch, size[0]//5)
        self.category = 12
        self.Object_surface_ons = [torch.zeros(size), torch.zeros(size), torch.zeros(size)]
        self.Object_surface_offs = [torch.zeros(size), torch.zeros(size), torch.zeros(size)]
        self.Eij = torch.zeros(size)
        self.Y_ij = torch.zeros(size)
        self.Cij = torch.zeros(1, 1, size[0], size[1])
        self.Sijf = torch.zeros(size)
        self.B = torch.zeros(size)
        self.W = torch.ones( (self.category, 20000) )
        self.fuzzy_ART = fuzzy_ART(X_size=100 * 100, c_max=self.category, rho=0.85, alpha=0.00001, beta=1)
        self.t = t
        self.Vjq = torch.zeros( (self.batch, self.category) )

        self.t_view = [{"Vjiq": torch.zeros( (self.batch, self.category) ),
                        "Oj": torch.zeros( (self.category) ), 
                        "Fj": torch.zeros( (self.category) ), 
                        "Dj": torch.zeros( (self.category) ), 
                        "Nj": torch.zeros( (self.category) )},
                       {"Vjiq": torch.zeros( (self.batch, self.category) ),
                        "Oj": torch.zeros( (self.category) ), 
                        "Fj": torch.zeros( (self.category) ), 
                        "Dj": torch.zeros( (self.category) ), 
                        "Nj": torch.zeros( (self.category) )}]

        #up
        self.W_vo = torch.ones( (self.batch, self.category, self.category) )
        self.W_ov = torch.ones( (self.batch, self.category, self.category) )

        self.W_od = torch.ones( (self.category, self.category) )
        self.W_df = torch.ones( (self.category, self.category) )

        self.W_fn = torch.ones( (self.category, self.category) )
        self.W_nf = torch.ones( (self.category, self.category) )

        self.W_of = torch.ones( (self.category) )        
        self.W_fo = torch.ones( (self.category) )

        self.W_ve = torch.ones_like((self.Eij))  #****
        self.EIJ = torch.zeros_like(self.Eij)
        self.Amn = torch.zeros(1, 1, size[0], size[1])
        self.Y_ijE = torch.zeros(size)
        self.Y_ijA = torch.zeros(size)

        self.Rwhere = Rwhere
        self.yR = 0
        self.key = key
        self.G = self.get_G(self.key)
        self.Uj = 0
        self.Tj = 0
        self.Pj = 0
        self.max_place = [249, 249]
        self.K_e = 10**-7

    def update(self, value):
        self.t_view[0] = self.t_view[1]
        self.t_view[1] = value

    def part1(self, input):
        Z = input["Z"], 
        imgsc_on =  input["imgsc_on"]
        imgsc_off = input["imgsc_off"]
        # for i in range(20):
        self.B = self.Boundaries(Z[0])
        type_input(self.B, "B", 1)
        self.Object_surface_ons = self.Surface_filling_in_(self.B, self.Object_surface_ons, imgsc_on)
        type_input(self.Object_surface_ons, "Object_surface_ons", 1)
        self.Object_surface_offs = self.Surface_filling_in_(self.B, self.Object_surface_offs, imgsc_off)
        type_input(self.Object_surface_offs, "Object_surface_offs", 1)
        self.Cij = self.Surface_contours()
        type_input(self.Cij, "Cij", 1)
        self.S_ij = 0.05 * (self.Object_surface_ons[0] + self.Object_surface_offs[0]) + 0.1 * (self.Object_surface_ons[1] + self.Object_surface_offs[1]) + 0.85 * (self.Object_surface_ons[2] + self.Object_surface_offs[2])
        type_input(self.S_ij, "S_ij", 1)
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
        self.Vjq = self.normalized_V_(input)
        value = self.t_view[0]
        #Vjiq
        Vjiq = (-0.01 * value["Vjiq"] + self.t * ((1 + torch.sum(self.W_ov * signal_m_function(value["Oj"]))) * (F.relu(self.Vjq) + self.G)) - self.Rwhere) * self.dt + value["Vjiq"]
        
        #A63
        Oj = value['Oj'] + self.dt * (-value['Oj'] + (1 + 2 * value['Fj'] * self.W_fo) * (0.5 * self.neuron3d(signal_m_function(F.relu(value['Vjiq'])), self.W_vo)  + self.G) - self.Rwhere)
        Oj = self.normalized_O_(Oj)
        
        #A66
        Dj = -value['Dj'] + self.Uj + self.Tj + 0.1 * self.neuron1d(signal_m_function(value['Oj']), self.W_od)   #待修改
        Dj = self.normalized_D_(Dj)

        Fj = (-value['Fj'] + (0.5 * signal_m_function(value['Oj']) * self.W_of + self.G) * (1 + self.neuron1d(value['Dj'], self.W_df) + self.neuron1d(signal_m_function(value['Nj']), self.W_nf))) * self.dt * 20 + value['Fj']
        Fj = self.normalized_F_(Fj)

        Nj = (- value['Nj'] + self.neuron1d(signal_m_function(value['Fj']), self.W_fn) + self.Pj) * 20 * self.dt
        Nj = self.normalized_N_(Nj)
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

    # def part3(self):
    #     for i in range(40):
    #         self.Eye_movements_map()
    #     self.Gain_field()
    #     self.attention_shroud()
    #     self.Reset()

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
            sumve = 0
            for i in range(self.Vjq.shape[0]):
                sumve += (Vjq[i]* (self.Eij - self.W_ve).unsqueeze(dim=2)).sum()
                self.W_ve = sumve * h_Eij * 500 * self.dt + self.W_ve
        
        sumvw = 0
        for i in range(self.Vjq.shape[0]):
            sumvw += (Vjq[i]*self.W_ve.unsqueeze(dim=2)).sum()

        Eij = F.relu(self.Cij)+625*inter_value_J + sumvw
        Eij = self.Y_ijE*(1-self.Eij.unsqueeze(dim=0).unsqueeze(dim=0))*Eij
        Eij = Eij - 0.01 * self.Eij * torch.sum(F.relu(self.Cij) + inter_value_K)
        Eij = Eij-self.Eij
        self.Eij = (Eij * self.dt + self.Eij).squeeze(dim=0).squeeze(dim=0)

        max_value = torch.max(self.Eij)

        if max_value>0.58:  #0.58
            self.max_place = (self.Eij==torch.max(self.Eij)).nonzero()[0]
            self.Eij[self.max_place[0]][self.max_place[1]] = max_value

    def shifted_map(self, input, key):
        B, C, H, W = input.shape
        #定义扩张后的眼球map
        c=torch.zeros((B, C, H*2, W*2))
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
            sumvb += torch.sum(Vjq[i]*self.W )

        output = []
        for i in range(len(Z_complex_cells)):
            #A20
            Gs_F = gaussianKernel(ksize=35, sigma=5, c=1/(2*np.pi * 25))

            C_S2d = gaussianconv_2d(self.Cij, Gs_F, 5) 
            if i == 2:
                B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d + sumvb) + 0.4 * self.Cij.sum()
                B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d + sumvb)  #- 0.4 * C_Surface_contours.sum()
                B = F.relu(B_molecular/B_denominator)
                output.append(B)
            else:
                B_denominator = 0.1 + Z_complex_cells[i] * (1 + 10**4 * C_S2d) + 0.4 * self.Cij.sum()
                B_molecular = Z_complex_cells[i] * (1 + 10**4 * C_S2d) - 0.4 * self.Cij.sum()
                B = F.relu(B_molecular/B_denominator)
                output.append(B)
        return output
 


def main():
    global argument
    argument = {}

    img = data_process("/Users/leilei/Desktop/ARTSCAN/output.png")
    #A.1. Retina and LGN cells
    imgsc_on, imgsc_off = GaussianBlur(img, [5,17,41], sigmac = [0.3, 0.75, 2], sigmas = [1, 3, 7])
    #A.2. V1 polarity-sensitive oriented simple cells
    # model = fuzzy_ART(X_size=100 * 100, c_max=100, rho=0.85, alpha=0.00001, beta=1)
    #A13
    Y_ons, Y_offs = Gabor_conv(imgsc_on, imgsc_off, sigmav= [3, 4.5, 6], sigmah = [1, 1.5, 2], Lambda = [3, 5, 7], angles = [0, 45, 90, 135], K_size = [(19,19), (29,29), (39,39)])
    #A16
    Z = complex_cells(Y_ons, Y_offs)
    type_input(Z, "Z", 1)
    argument["Z"] = Z 
    argument["imgsc_on"] = imgsc_on
    argument["imgsc_off"] = imgsc_off
    model1 = arcscan(size=(500,500))
    eye_path = cv2.imread("/Users/leilei/Desktop/ARTSCAN/output.png")
    max_index = [(int(model1.max_place[0]), int(model1.max_place[1]))]
    for t in range(25):
        model1.train(argument)
        max_index.append((int(model1.max_place[0]), int(model1.max_place[1])))
        img1 = cv2.circle(eye_path,(int(model1.max_place[0]), int(model1.max_place[1])),5,(0,0,255),-1)
        img1 = cv2.line(img1, max_index[t+1], max_index[t], (0, 255, 0), 2)
        cv2.imwrite("/Users/leilei/Desktop/ARTSCAN/output_path.png", img1)
if __name__ == "__main__":
    main()

