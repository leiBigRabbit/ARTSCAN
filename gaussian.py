import cv2
import numpy as np

# 加载图像
img = cv2.imread('image.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存灰度图
cv2.imwrite('output.jpg', gray_img)
# 定义高斯卷积核大小
kernel_sizes = [(5, 5), (41, 41), (17, 17)]

# 对图像进行不同大小的高斯卷积
for kernel_size in kernel_sizes:
    # 生成高斯卷积核
    kernel = cv2.getGaussianKernel(kernel_size[0], -1) * cv2.getGaussianKernel(kernel_size[1], -1).T
    # 进行卷积操作
    img_blur = cv2.filter2D(gray_img, -1, kernel)
    # 显示结果
    save_img = 'Gaussian Blur ({}, {})'.format(kernel_size[0], kernel_size[1]) + '.jpg'
    cv2.imwrite(save_img, img_blur)
    # cv2.imshow('Gaussian Blur ({}, {})'.format(kernel_size[0], kernel_size[1]), img_blur)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
