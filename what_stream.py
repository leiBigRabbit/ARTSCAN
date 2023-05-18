import torch
import numpy as np
# from fuzzy_ART import fuzzy_ART
from utils.utils import half_rectified, type_input, data_process
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
    return argument