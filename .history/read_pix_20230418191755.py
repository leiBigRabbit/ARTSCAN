# import cv2
# img = cv2.imread("output4.jpg")
# h,w,_ = img.shape
# for i in range(h):
#     for j in range(w):
#         if img[i,j].sum()>0:
#             print(img[i,j])
# # print(h,w)


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
img = np.zeros((3,3), dtype=np.uint8)
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = np.array(randomByteArray)
# 转换为灰色图像
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite("RondomGray.png", grayImage)
# 转换为BGR图像
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite("RandomColor.png", bgrImage)