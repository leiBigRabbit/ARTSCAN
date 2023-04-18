import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 定义高斯内核大小和标准差
kernel_size = (5, 5)
sigma = 1.0

img_gaussian = cv2.GaussianBlur(img, kernel_size, sigma)

cv2.imshow('Gaussian Image', img_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
