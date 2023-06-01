import os
import glob
from PIL import Image
import cv2
import random

def pad_img(img_path,w_=1000,h_=1000,value_=[128,128,128]):
    """args:
    img_: opencv的img\n
    w_: 宽
    h_: 高
    value: padding的值
    return: 上下左右各padding之后的opencv的img
    """
    img = cv2.imread(new_image_path)

    h,w,_ = img.shape
    top = (h_ - h ) // 2
    bottom = (h_ - h ) - top
    left = (w_ - w) // 2
    right = (w_ - w) - left
    # value_ = [img_[0][0][0], img_[0][0][0], img_[0][0][0]]
    pad_img = cv2.copyMakeBorder(img,
                top=top,bottom=bottom,
                left=left,right=right,
                borderType=cv2.BORDER_CONSTANT,
                value=value_)
    
    return pad_img


def cat_img(image_files, image_dir, size):
 
    # 加载图像，调整大小
    images = []
    for image_file in image_files:
        image = Image.open(image_file)
        image = image.resize(size)
        images.append(image)
    
    # 创建一个新的500x500像素大小的灰色背景图像
    new_image = Image.new('RGB', (500, 500), 'grey')
    
    index = [i for i in range(25)]
    if len(images) < 25:
        index = random.sample(index, len(images))

    for ind in range(len(index)):
        new_image.paste(images[ind], (int(index[ind]%5)*100+20, int(index[ind]/5)*100+20))

    # 按顺序将25个图像粘贴到新图像的正确位置
    # ind = 0
    # for i in range(5):
    #     for j in range(5):
    #         new_image.paste(images[ind], (i*100+20, j*100+20))
    #         ind+=1

    new_image.save('./output.png')
    return  './output.png'     
    

if __name__=='__main__':

    #合成图片
    # 设置图像文件夹的路径
    image_dir = './img/'
    # 获取文件夹中所有图像文件的列表
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    size = (50,50)
    new_image_path = cat_img(image_files, image_dir, size)

    img_pad = pad_img(new_image_path,)
    cv2.imwrite("./output_pad.jpg",img_pad)
    # cv2.imshow('pad img',img_pad)
    # cv2.waitKey(0)

