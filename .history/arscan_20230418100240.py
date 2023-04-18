import cv2
import numpy as np
from PIL import Image, ImageFilter

# 加载图像
img = Image.open('image.jpg')

# 定义高斯卷积核大小
kernel_sizes = [(3, 3), (5, 5), (7, 7)]

# 对图像进行不同大小的高斯卷积
for kernel_size in kernel_sizes:
    # 生成高斯卷积核
    kernel = ImageFilter.Kernel(kernel_size, np.array(cv2.getGaussianKernel(kernel_size[0], -1) * cv2.getGaussianKernel(kernel_size[1], -1).T))
    print(kernel)
    # 进行卷积操作
    img_blur = img.filter(kernel)
    # 显示结果
    img_blur.show()

