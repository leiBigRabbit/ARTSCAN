import torch
import numpy as np
from fuzzy_ART import fuzzy_ART
#A54/53
def Vector_Bq(input):
    uboundary_on = torch.where(input <= 0, 0, 1)
    uboundary_off = 1 - input
    uboundary_on = uboundary_on.reshape(1, 25, 100, 100)
    uboundary_off = uboundary_off.reshape(1, 25, 100, 100)
    input = input.reshape(1, 25, 100, 100)
    output = input * uboundary_on, uboundary_on * uboundary_off
    output = torch.cat(output,dim=2).reshape(1, 25, 2, 100,100)
    return output
# model = fuzzy_ART(x_size=(100,100), c_max=100, rho=0.85, alpha=0.00001, beta=1)
