import os
import glob
from PIL import Image
 
# 设置图像文件夹的路径
image_dir = '/Users/leilei/Desktop/img/'
 
# 获取文件夹中所有图像文件的列表
image_files = glob.glob(os.path.join(image_dir, '*.png'))
 
# 加载每个图像，并调整为200x200像素大小（如果需要）
images = []
for image_file in image_files:
    image = Image.open(image_file)
    image = image.resize((50, 50))
    images.append(image)
 
# 创建一个新的400x400像素大小的白色背景图像
new_image = Image.new('RGB', (500, 500), 'grey')
 
# 将四个图像粘贴到新图像的正确位置
ind = 0
for i in range(5):
    for j in range(5):
        new_image.paste(images[ind], (i*100+20, j*100+20))
        ind+=1
        
        
# 将最终图像保存到磁盘上
new_image.save(image_dir + 'output.png')
