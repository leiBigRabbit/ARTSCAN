import torch
import numpy as np
from fuzzy_ART import fuzzy_ART
from utils.utils import half_rectified, type_input, data_process
#A54/53
def Vector_Bq(input):
    # input = torch.where(input <= 0, 0, 1)  #A54
    input = torch.cat(torch.split(input[0][0], 100, 0),dim=1)
    output = torch.split(input, 100, 1)
    output = torch.cat([fm.unsqueeze(0).reshape(-1,10000) for fm in output], dim=0)

    # input_u = torch.where(input <= 0, 0, 1)
    # input_u = torch.split(input_u, 100, 1)
    # input_u = torch.cat([fm.unsqueeze(0).reshape(fm.shape[0]*fm.shape[1]) for fm in input_u], dim=0).squeeze(0)

    # output = input
    return output

def activity_V(Bq, Wq, sigma, V_jq):
    Wq = torch.cat(Wq, 1-Wq)
    Bq = torch.sum(torch.minimum(Bq, Wq))
    Vjq = (1 + 0.1 * V_jq) * Bq / (sigma + Wq)
    return Vjq