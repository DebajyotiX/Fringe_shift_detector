import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks

# Load the "seemingly colorless but actually in RGB space" image in grayscale
img = cv2.imread('ccd_image_compressed.png',0)

# Denoising
denoised = cv2.GaussianBlur(img,(5,5),0)
for i in range(30):
    denoised = cv2.GaussianBlur(denoised,(5,5),0)

# Seperating the two parts
upper_half_image = denoised[0:143,:]
lower_half_image = denoised[144:-1,:]

# Displaying parts
cv2.imshow('TOP PART',upper_half_image)
cv2.imshow('BOTTOM PART',lower_half_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

lower_fringe_ = lower_half_image[23,:]
print(np.shape(lower))
x=np.array(range(536))
# print(np.shape(x))
plt.plot(x,lower)
plt.xlabel("Pixel number")
plt.ylabel("Relative brightness")

#-----------peaks----------------#
peaks, properties = find_peaks(lower,height=5, width=3)
plt.plot(x[peaks], lower[peaks], "o", color='#ff6400')
#-----------peaks----------------#

plt.show()

# plt.imshow(blur, cmap = 'gray', interpolation = 'bicubic')
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()