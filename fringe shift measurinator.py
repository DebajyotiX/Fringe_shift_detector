import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks

# Load the "seemingly colorless but actually in RGB space" image in grayscale
img = cv2.imread('ccd_image_compressed.png',0)

# Denoising
denoised = cv2.GaussianBlur(img,(5,5),0)
for i in range(30):   #I blur is 30 times. Its the sweet spot
    denoised = cv2.GaussianBlur(denoised,(5,5),0)

# Seperating the two parts
upper_half_image = denoised[0:143,:]
lower_half_image = denoised[144:-1,:]

# Displaying parts
cv2.imshow('TOP PART',upper_half_image)
cv2.imshow('BOTTOM PART',lower_half_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Calculation of the peaks,
 by choosing two pixel line from the two parts of the image
 this pixel lines runs horizontally and cuts the fringes perpendicularly
'''

lower_fringe_linear_sample = lower_half_image[23,:]
upper_fringe_linear_sample = upper_half_image[123,:]
x_pix_array = np.array(range(536))
zero = np.zeros(536) # Zero array for plot purpose
peaks, properties = find_peaks(lower_fringe_linear_sample,height=5, width=3)
peaks2, properties2 = find_peaks(upper_fringe_linear_sample,height=5, width=3)

'''
Here I will find out the wavelength in units of pixels.
I will calculate it from pixel to pixel distances of consecutive peaks
'''


#---------------- PLOT --------------------#
plt.rcParams["figure.figsize"] = [10,5]
fig, ax = plt.subplots(nrows=1, ncols=2,)
ax[0].plot(x_pix_array,upper_fringe_linear_sample)
ax[0].plot(x_pix_array[peaks2], upper_fringe_linear_sample[peaks2], "o", color='#ff6400',label="Peak")
ax[0].plot(x_pix_array[peaks2], zero[peaks2], "o", color='#ff6400')
ax[0].set_xlabel("Pixel number")
ax[0].set_ylabel("Relative brightness")
ax[0].set_title("Top Interference pattern")
ax[0].legend()

ax[1].plot(x_pix_array,lower_fringe_linear_sample)
ax[1].plot(x_pix_array[peaks], lower_fringe_linear_sample[peaks], "o", color='#ff6400',label="Peak")
ax[1].plot(x_pix_array[peaks], zero[peaks], "o", color='#ff6400')
ax[1].set_xlabel("Pixel number")
ax[1].set_ylabel("Relative brightness")
ax[1].set_title("Bottom Interference pattern")
ax[1].legend()
plt.show()
#---------------- PLOT OVER --------------------#








