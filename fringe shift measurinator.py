import matplotlib.pyplot as plt
import numpy as np
import cv2
# import time
# import sys

# Load an color image in grayscale
img = cv2.imread('ccd_image_compressed.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)
for i in range(30):
    blur = cv2.GaussianBlur(blur,(5,5),0)

# a=np.max(blur)
# blur=blur*128/a
half1=blur[0:143,:]
half2=blur[144:-1,:]
# cv2.imshow('image upper',half1)
cv2.imshow('image lower',half2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

lower = half2[66,:]
print(np.shape(lower))
x=np.array(range(536))
# print(np.shape(x))
plt.plot(x,lower)
plt.show()
# plt.imshow(blur, cmap = 'gray', interpolation = 'bicubic')
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()