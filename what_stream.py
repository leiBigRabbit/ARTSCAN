import torch
import numpy as np
from fuzzy_ART import fuzzy_ART
from utils.utils import half_rectified, type_input, data_process
from utils.utils import half_rectified_relu, signal_m_function
import torch.nn.functional as F

class what_stream:
    def __init__(self, batch, size, Rwhere, t = 0.6, dt = 0.05, key='up'):
        self.batch = batch
        self.size = size
        self.W_vo = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_od = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_df = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_fn = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_of = torch.ones( (self.batch, self.size) ).to(self.device)

        self.W_nf = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_fo = torch.ones( (self.batch, self.size) ).to(self.device)
        self.W_ov = torch.ones( (self.batch, self.size) ).to(self.device)

        self.Vjiq = torch.zeros( (self.batch, self.size) ).to(self.device)
        self.Oj = torch.zeros( (self.batch, self.size) ).to(self.device)
        self.Fj = torch.zeros( (self.batch, self.size) ).to(self.device)
        self.Dj = torch.zeros( (self.batch, self.size) ).to(self.device)
        self.Nj = torch.zeros( (self.batch, self.size) ).to(self.device)

        self.Rwhere = Rwhere
        self.t = t
        self.dt = dt
        self.key = key
        self.G = self.get_G(self.key)
    
    def get_G(self, key):
        if key == "down":
            return 0.1
        else:
            return 0
    #A61    
    def build_View_category_integrator(self, Vjq):
        try:
            self.Vjiq = (-0.01 * self.Vjiq + self.t * (1 + torch.sum(signal_m_function(self.Oi) * self.W_ov)) * (half_rectified(Vjq) + self.G) - self.Rwhere) * self.dt + self.Vjiq
        except:
            self.Vjiq = (self.t * (Vjq + self.G) - self.Rwhere) * self.dt
        return self.Vjiq

    def build_Invariant_object_category(self, Vjiq):
        try:
            self.Oj = self.Oj + self.dt * (-self.Oj + (1 + 2 * self.Fj * self.W_fo) * (0.5 * torch.sum(signal_m_function(half_rectified_relu(Vjiq))* self.W_vo)  + self.G) - self.Rwhere)
        except:
            self.Oj = 0.5 * torch.sum(signal_m_function(half_rectified_relu(self.Vjiq))) * self.W_vo
        
        #A64
        self.Oj = self.Oj_normalized(self.Oj)
        # argument['Wvo'] = weight_Wvo(argument)
        return self.Oj     
       
    def train(self, Vjq):
        input = self.build_View_category_integrator(Vjq)
        input = self.build_Invariant_object_category(input)
        return input

    def Oj_normalized(self, Oj):
        Oj_on = F.relu(Oj)
        Oj_on, _ = Oj_on.sort(descending=True)
        min_value = Oj_on[:,-1:]
        max_value = Oj_on[:,:1]

        Oj_on = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), Oj_on), dim=1)
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


#A54/53
def Vector_Bq(argument):
    input = argument["Boundary"]
    # input = torch.where(input <= 0, 0, 1)  #A54
    input = torch.cat(torch.split(input[0][0], 100, 0),dim=1)
    output = torch.split(input, 100, 1)
    output = torch.cat([fm.unsqueeze(0).reshape(-1,10000) for fm in output], dim=0)

    # input_u = torch.where(input <= 0, 0, 1)
    # input_u = torch.split(input_u, 100, 1)
    # input_u = torch.cat([fm.unsqueeze(0).reshape(fm.shape[0]*fm.shape[1]) for fm in input_u], dim=0).squeeze(0)

    # output = input
    argument["B_q"] = output
    return argument


#A60
def activity_V(argument):
    # argument["V"][0][1] = 1.104 #example
    Vjq_value, _ = argument["V"].sort(descending=True)
    min_value = Vjq_value[:,-1:]
    max_value = Vjq_value[:,:1]
    Vjq_value = torch.cat((torch.arange(0, 25, 1).unsqueeze(dim = 1), Vjq_value), dim=1)  #增加一列index列
    key = torch.sum(((max_value - (Vjq_value[:,2:] + 0.03)) > 0), dim=1) == (Vjq_value.shape[1]-2)
    # key = key.expand(100,25).transpose(0,1)
    #A60/1
    part1 = Vjq_value[key]  
    #A60/2
    key2 = max_value[~key].squeeze(dim=1) - min_value[~key].squeeze(dim=1) < 0.03 #实验数据设置0.5546
    part2 = Vjq_value[~key][key2]
    part2[:,1:] = part2[:,1:] * (part2[:,1:] / torch.sum(part2[:,1:]))
    
    part3 = Vjq_value[~key][~key2]
    part3[:,1:] = part3[:,1:] * 0
    Vjq = torch.cat((part1,part2,part3),dim=0)
    # torch.masked_select(Vjq_value,key.expand(100,25).transpose(0,1))
    argument["Vjq"] = sorted(Vjq, key = lambda x:x[0])
    argument["Vjq"] = torch.tensor([item.cpu().detach().numpy() for item in argument["Vjq"]])[:,1:]
    return argument