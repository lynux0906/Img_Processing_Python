import cv2
import numpy as np
from matplotlib import pyplot as plt 

class threshold:
	def __init__(self, src, value1, value2):
		self.image = src
		self.thresh_value = value1
		self.max_value = value2

	def thresh_binary(self):
		w, h = self.image.shape
		dst = self.image.copy()
		for i in range(w):
			for j in range(h):
				if(self.image[i, j] >= self.thresh_value):
					dst[i, j] = self.max_value
				else:
					dst[i, j] = 0
		return dst

	def thresh_binary_inv(self):
		w, h = self.image.shape
		dst = self.image.copy()
		for i in range(w):
			for j in range(h):
				if(self.image[i, j] >= self.thresh_value):
					dst[i, j] = 0
				else:
					dst[i, j] = self.max_value
		return dst

	def thresh_trunc(self):
		w, h = self.image.shape
		dst = self.image.copy()
		for i in range(w):
			for j in range(h):
				if(self.image[i, j] >= self.thresh_value):
					dst[i, j] = self.max_value
		return dst

	def thresh_tozero(self):
		w, h = self.image.shape
		dst = self.image.copy()
		for i in range(w):
			for j in range(h):
				if(self.image[i, j] < self.thresh_value):
					dst[i,j] = 0
		return dst

	def thresh_tozero_inv(self):
		w, h = self.image.shape
		dst = self.image.copy()
		for i in range(w):
			for j in range(h):
				if(self.image[i, j] >= self.thresh_value):
					dst[i, j] = 0
		return dst

img = cv2.imread('messi.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = threshold(img_gray, 100, 255).thresh_binary()
thresh_inv = threshold(img_gray, 100, 255).thresh_binary_inv()
thresh_trun = threshold(img_gray, 100, 255).thresh_trunc()
thresh_to0 = threshold(img_gray, 100, 255).thresh_tozero()
thresh_to0_inv = threshold(img_gray, 100, 255).thresh_tozero_inv()

titles = ["Original Image", "Thresh Binary", "Thresh Binary Inverse",
			"Thresh Trunc", "Thresh To Zero", "Thresh To Zero Inverse"]

images = [img, thresh, thresh_inv, thresh_trun,
			thresh_to0, thresh_to0_inv]

for i in range(6):
	plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])

plt.show()