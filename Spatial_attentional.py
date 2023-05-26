from utils.kernel import gaussianKernel
from utils.conv2d import gaussianconv_2d
from utils.utils import signal_f_function, signal_m_function, signal_g_function
import torch.nn.functional as F
import torch.nn as nn

# class attentional(nn.modules):
#     def __init__(self, ksize):
#         super(attentional, self).__init__()
#         self.f = signal_f_function()
#         self.g = signal_g_function()
#         self.m = signal_m_function()
#         self.ksize = ksize
#         self.kernel_C = gaussianKernel(self.ksize, 4, 0.1)
#         self.kernel_E = gaussianKernel(self.ksize, 200, 1/(10**5 * 2))
#         self.dt = 0.25
#     def forward(self, input, y_ij):
#         gA = self.g(input)
#         fA = self.f(input)
#         input_A = 1 + 0.2 * gaussianconv_2d(fA, self.kernel_C, self.ksize)
#         #*****
#         second = -input * (self.g(input))
#         input_A = (input_A * gA * (1-input) - 0.1 * input) * habituative_transmitter(input, y_ij ,self.dt)
#         output = input_A + second
#         return output

#A42
def habituative_transmitter(input, y_ij, dt, AImn):
    kernel_C = gaussianKernel(3, 4, 0.1)
    gA = signal_g_function(AImn)
    fA = signal_f_function(input)
    fA = gaussianconv_2d(fA, kernel_C, 3)
    input_A = 3 * 10**6 * gA * (1 + 0.2 * fA) * y_ij
    output = 10*-8 * (2 - y_ij - input_A) * dt + y_ij
    return output

#A37
def attention_shroud(input, y_ij, AImn, dt = 0.05):
    rwhere = Reset(input)
    kernel_C = gaussianKernel(3, 4, 0.1)
    kernel_E = gaussianKernel(3, 200, 1/(10**5 * 2))
    gA = signal_g_function(AImn)
    fA = signal_f_function(input)
    input_A = gA * (1 + 0.2 * gaussianconv_2d(fA, kernel_C, 3)) * (1 - input) - 0.1*input
    #*****
    yAij = habituative_transmitter(input, y_ij ,dt, AImn)
    second = -input * gaussianconv_2d((gA + fA), kernel_E, 3) + 10 * rwhere
    output = (yAij + second) * 10 * dt + input
    return output
