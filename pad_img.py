import cv2


def pad_img(img_,w_=1000,h_=1000,value_=[128,128,128]):
    """args:
    img_: opencv的img\n
    w_: 宽
    h_: 高
    value: padding的值
    return: 上下左右各padding之后的opencv的img
    """
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


filename = "/Users/leilei/Desktop/ARTSCAN/output.png"
    
img = cv2.imread(filename)
img_pad = pad_img(img,)
cv2.imwrite("2.jpg",img_pad)
# cv2.imshow('pad img',img_pad)
# cv2.waitKey(0)
