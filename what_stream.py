import torch
import numpy as np

#A54/53
def Vector_Bq(input):
    uboundary_on = torch.where(input <= 0, 0, 1)
    uboundary_off = 1 - input
    uboundary_on = uboundary_on.reshape(1, 25, 100, 100)
    uboundary_off = uboundary_off.reshape(1, 25, 100, 100)
    input = input.reshape(1, 25, 100, 100)
    output = (input * uboundary_on, uboundary_on * uboundary_off)
    return output