import cv2
img = cv2.imread("output4.jpg")
h,w,_ = img.shape
for i in range(h):
    for j in range(w):
        print(img[i,j])
        exit()
print(h,w)