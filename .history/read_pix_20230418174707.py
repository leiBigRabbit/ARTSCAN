import cv2
img = cv2.imread("output4.jpg")
h,w,_ = img.shape
for i in range(h):
    for j in range(w):
        if img[i,j] == [0,0,0]:
            print(img[i,j])
print(h,w)