import cv2
import numpy as np 

def addWeighted(image1, image2, rate):
    w, h, _ = image1.shape
    dst = image1.copy()
    for i in range(w):
        for j in range(h):
            dst[i, j] = image1[i, j] * rate[0] + image2[i, j] * rate[1]

    return dst

img1 = cv2.imread("/home/levanlinh/Pictures/Opencv/lena.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("/home/levanlinh/Pictures/Opencv/opencv_logo.png", cv2.IMREAD_COLOR)

img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

blending = addWeighted(img1, img2, [0.7, 0.3])
blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
print(blending[0, 0])

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Blending", blending)
cv2.imshow("Blend", blend)

cv2.waitKey(0)
cv2.destroyAllWindows()
