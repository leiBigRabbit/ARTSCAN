from utils.utils import half_rectified_relu, signal_m_function
import torch.nn.functional as F

import torch 

def weight_Wvo(argument):
    try:
        Wvo = 0.003 * (1 - argument["Wvo"]) * F.relu(argument["Vjiq"]) * F.sigmoid(argument["Oj"]) - 0.001*argument["Wvo"]*F.relu(argument["Vjiq"])*torch.sum(argument["Oj"])  #待补充******
        argument["Wvo"] = Wvo
    except:
        argument["Wvo"] = torch.ones_like(argument["Vjiq"])
        # Wvo = 0.003 * F.relu(argument["Vjiq"]) * signal_m_function(0)
        # argument["Wvo"] = Wvo
    return argument

#A64
def Oj_normalized(Oj):
    
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

#A63
def object_category_neuron(argument):
    G = argument["G"]
    dt = argument["dt"]
    Rwhere = argument["Rwhere"]
    Vjiq = argument["Vjiq"]
    
    try:
        Fj = argument['Fj']
        Wfo = argument['Wfo']
        Oj = argument['Oj']
        Wvo = argument['Wvo']
        Oj = Oj + dt * (-Oj + (1 + 2 * Fj * Wfo) * (0.5 * torch.sum(signal_m_function(half_rectified_relu(Vjiq))* Wvo)  + G) - Rwhere)
    except:
        argument['Wvo'] = torch.ones_like(Vjiq)
        Oj = 0.5 * torch.sum(signal_m_function(half_rectified_relu(Vjiq))) * argument['Wvo']
    
    #A64
    Oj = Oj_normalized(Oj)
    argument['Oj'] = Oj
    argument['Wvo'] = weight_Wvo(argument)
    # if argument['t'] == 1:
    #     Oj, Wfo, F, Wvo = 0, 0, 0, 0
    # else:
    #     Fj = argument['Fj']
    #     Wfo = argument['Wfo']
    #     Oj = argument['Oj']
    #     Wvo = argument['Wvo']
    # argument['Wfo'] = signal_m_function(Fj) * signal_m_function(Oj) * (signal_m_function(Oj) - Wfo)
    # Oj = Oj + dt * (-Oj + (1 + 2 * Fj * Wfo) * (0.5 * torch.sum(signal_m_function(half_rectified_relu(Vjiq))* Wvo)  + G) - Rwhere)
    
    return argument