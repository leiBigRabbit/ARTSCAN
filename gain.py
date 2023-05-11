from utils.kernel import gaussianKernel
import numpy as np
# def  activity_I(S_input, )

def shifted_map(input, max_place, key):
    B, C, H, W = input.shape
    #定义扩张后的眼球map
    c=np.zeros((B, C, H*3, W*3))
    #赋值0.5后嵌入原S的视图
    c[:] = 0.5
    c[:, :, H:2*H, W:2*W] = input
    #中间位置y,x平移(k,l)个单位A32中的S(m-k,n-l)
    if key:
        output = c[:, :, H - max_place[2]:2*H - max_place[2], W - max_place[3]: 2*W - max_place[3]]
    else:
        output = c[:, :, H + max_place[2]:2*H + max_place[2], W + max_place[3]: 2*W + max_place[3]]
    return output

# Gain field
def Gain_field(S_input, max_place, Amn):
    S_eye = shifted_map(S_input, max_place, 1)
    shifted_A = shifted_map(Amn, max_place, 0)
    #A34--AImn
    output_A = S_eye + Amn
    #A36--SFmn
    output_S = shifted_A + S_input
    return output_A, output_S










# def gain_field():