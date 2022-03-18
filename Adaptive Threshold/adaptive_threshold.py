import cv2
import numpy as np
from matplotlib import pyplot as plt

class adaptivethreshold:
	def  __init__(self, input_image, value1, c):
		self.image = input_image
		self.h, self.w = self.image.shape
		self.max_value = value1
		self.c = c

	def adaptive_mean_c(self):
		dst = self.image.copy()
		#Create extend image
		extend_img = np.array(dst)
		a = []
		for i in range(dst.shape[1]):
			a = np.append(a, 0)
		b = np.array([a])
		extend_img = np.append(extend_img, b, axis = 0)
		extend_img = np.append(b, extend_img, axis = 0)
		c = np.array([[0]])
		for i in range (extend_img.shape[0] - 1):
			c = np.append(c, np.array([[0]]), axis = 0)
		# print(c.shape)
		extend_img = np.append(extend_img, c, axis = 1)
		extend_img = np.append(c, extend_img, axis = 1)
		#Chuyen vi ma tran
		# extend_img = extend_img.transpose()
		# print(extend_img)
		print(extend_img.shape)
		# print(self.w, self.h)
		print(dst.shape[0], dst.shape[1])

		for i in range(dst.shape[0]):
			for j in range(dst.shape[1]):
				# print(i, j)
				# print(extend_img[i, j+1])
				thresh = (extend_img[i, j+1] + extend_img[i+2, j+1] + extend_img[i+1, j] + extend_img[i+1, j+2])/4 - self.c
				# print(thresh)
				if(dst[i, j] > thresh):
					dst[i, j] = self.max_value
				else:
					dst[i, j] = 0
		return dst

	def adaptive_gaussian(self):
		gauss = 0.96865465464
		dst = self.image.copy()
		#Create extend image
		extend_img = np.array(dst)
		a = []
		for i in range(dst.shape[1]):
			a = np.append(a, 0)
		b = np.array([a])
		extend_img = np.append(extend_img, b, axis = 0)
		extend_img = np.append(b, extend_img, axis = 0)
		c = np.array([[0]])
		for i in range (extend_img.shape[0] - 1):
			c = np.append(c, np.array([[0]]), axis = 0)
		# print(c.shape)
		extend_img = np.append(extend_img, c, axis = 1)
		extend_img = np.append(c, extend_img, axis = 1)
		#Chuyen vi ma tran
		# extend_img = extend_img.transpose()
		# print(extend_img)
		print(extend_img.shape)
		# print(self.w, self.h)
		print(dst.shape[0], dst.shape[1])

		for i in range(dst.shape[0]):
			for j in range(dst.shape[1]):
				# print(i, j)
				# print(extend_img[i, j+1])
				thresh = (extend_img[i, j+1] + extend_img[i+2, j+1] + extend_img[i+1, j] + extend_img[i+1, j+2])*gauss/4 - self.c
				# print(thresh)
				if(dst[i, j] > thresh):
					dst[i, j] = self.max_value
				else:
					dst[i, j] = 0
		return dst

img = cv2.imread("/home/levanlinh/Pictures/Opencv/sudoku.jpg", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thr1 = adaptivethreshold(img_gray, 255, 2).adaptive_mean_c()
thr2 = adaptivethreshold(img_gray, 255, 2).adaptive_gaussian()

images = [img, img_gray, thr1, thr2]
titles = ["Original Image", "Gray Image", "Adaptive Mean", "Adaptive Gaussian"]

for i in range(4):
	plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.xticks([]), plt.yticks([])

plt.show()