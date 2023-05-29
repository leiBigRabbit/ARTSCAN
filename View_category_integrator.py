from utils.utils import signal_m_function, half_rectified
import torch

def get_G(key):
    if key == "down":
        return 0.1
    else:
        return 0

#A61
def View_category_integrator(argument, t=0.6, key="up", dt = 0.05):
    argument["G"] = get_G(key)
    try:
        argument['Wov'] = torch.ones_like(argument["Vjq"])
        Vjiq = (-0.01 * argument["Vjiq"] + t * (1 + torch.sum(signal_m_function(argument["Oi"]) * argument["W_ov"])) * (half_rectified(argument["Vjq"]) + argument["G"]) - argument['Rwhere']) * argument["dt"] + argument["Vjiq"]
        argument["Vjiq"] = Vjiq
    except:
        # 
        Vjiq = (t * (argument["Vjq"] + argument["G"]) - argument['Rwhere']) * argument['dt']
        argument["Vjiq"] = Vjiq
    return argument
    # Vjq = argument["Vjq"]
    # # W_ov = argument["W_ov"]
    # if "Oi" not in argument:
    #     Oi = 0
    # else:
    #     Oi = argument["Oi"]
    # Rwhere = argument["Rwhere"]
    
    # if "Vjiq" not in argument:
    #     argument["Vjiq"] = 0
    # if "W_ov" not in argument:
    #     Vjiq = (-0.01 * argument["Vjiq"] + t * (half_rectified(Vjq) + argument["G"]) - Rwhere) * dt + argument["Vjiq"]
    # else:
    #     Vjiq = (-0.01 * argument["Vjiq"] + t * (1 + torch.sum(signal_m_function(Oi) * argument["W_ov"])) * (half_rectified(Vjq) + argument["G"]) - Rwhere) * dt + argument["Vjiq"]
    # argument["Vjiq"] = Vjiq
    # return argument

