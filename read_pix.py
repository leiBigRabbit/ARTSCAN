# import cv2
# img = cv2.imread("output4.jpg")
# h,w,_ = img.shape
# for i in range(h):
#     for j in range(w):
#         if img[i,j].sum()>0:
#             print(img[i,j])
# # print(h,w)
import numpy as np
import torch

a = 0.00001
b = 10000.0
c = torch.tensor([10000.0,10000.0])
print(c/(a+b))
exit()
A = torch.tensor([[1,2,4,-5,5,7],[4,-5,7,8,4,6],[1,2,-4,5,2,3],[4,5,7,-8,5,6],[4,-5,7,8,4,6],[1,2,-4,5,2,3]])
print(A)
A = A.reshape(2,2,2,-1)
print(A)
exit()
B = 1 - A
A1 = torch.where(A>0, 1, 0)

A = A.reshape(-1, 2, 2, 2)
A1 = A1.reshape(-1, 2, 2)
B = B.reshape(-1, 2, 2)

M = A * A1
N = B * A1
# mn = torch.hstack((M, N))  #.reshape(1,4,2,2,2)
MN = torch.cat((M,N),dim=2)
MN2 = torch.cat((M,N),dim=2).reshape(4,-1)
# k = torch.stack([M,N],dim=2).reshape(4,-1)
print(M)
print(N)
# print(mn.shape)
# print(mn)
print(MN)
print(MN2)
# print(k)
print(MN.shape)
# print(MN)
# print(MN.shape)
# print(mn)
# print(mn.shape)
# b = np.array([[1,2,3],[4,5,6],[1,2,3],[1,2,3],[4,5,6]])
# A = A.reshape(1, 4, 2, 2,order='C')
# print(A)
# reshaped = np.swapaxes(A.reshape(A.shape[0], A.shape[1]//3, 3, A.shape[1]//3, 3, order='C'), axis1=-2, axis2=-3)

# np.allclose(reshaped, An)  # This is true

# c =a * b
# print(c)

"""
c=np.zeros((1,1,9,9))
a = np.ones((1,1,3,3))
c[:] = 0.5
# print(c)
c[:,:,3:6,3:6] = a
c[:,:,3:4,3:4] = 2
c = torch.tensor(c)
print(c)
max_place = (c==torch.max(c)).nonzero()[0]
print(max_place)
max_place[2] = max_place[2] + 3
max_place[3] = max_place[3] + 3
print(max_place)
"""
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# img = np.zeros((3,3), dtype=np.uint8)
# randomByteArray = bytearray(os.urandom(120000))
# flatNumpyArray = np.array(randomByteArray)
# # 转换为灰色图像
# grayImage = flatNumpyArray.reshape(300, 400)
# cv2.imwrite("RondomGray.png", grayImage)

# # 转换为BGR图像
# bgrImage = flatNumpyArray.reshape(100, 400, 3)
# cv2.imwrite("RandomColor.png", bgrImage)