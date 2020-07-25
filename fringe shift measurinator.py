import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import sys


capture_frame = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
# put name of the video file inplce of 1.
while (True):
   start=time.time()    # Time stars now

   ret, frame = capture_frame.read()
   # out.write(frame) # for printing in a file
   #cv2.imshow('LIVE FEED-USB CAM',frame) -------------done using matplotlib
   frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

   # ------------Matrix calculations to get wavelength vs Intesity(realtive scale)
   pix_width,pix_height = 640,480
   grey = np.array(np.sum(frame, axis=2))
   grey = grey / 3  # Normalizing as I added the R, G and B values together
   line = np.array(np.sum(grey[100:200,:], axis=0))
   # line = np.array(np.sum(grey[150:160,:], axis=0))   # because of slanted grating

   # line = grey[,:]

   line = line /100   # Normalizing as I added a the column pixels
   lineflip = np.array(np.sum(grey, axis=0)) # initializing the array
   for i in range(pix_width):
       lineflip[pix_width - 1 - i] = line[i]
       # The array line is flipped horizontally to keep red on right side
   wavelength = np.array([(-.000001 * p ** 3 +.001 * p ** 2 + 0.9703 * p + 294.37) for p in range(1, pix_width + 1)])
   # wavelength = np.array([(.00001056 *(p ** 3)+.0052 * (p ** 2) + 2.0408 * p + 235.97) for p in range(1, pix_width + 1)])
   # ------------Matrix calculations ends--------------------------------

   # pixel no starts at 1 during calibration
   plt.close()
   # -------------------Fresh Plot from live-cam feed----------------

   fig, ax = plt.subplots(nrows=1, ncols=2,facecolor=(0.2, 0.2, 0.2))
   ax[0].set_facecolor((0.2, 0.2, 0.2))
   ax[0].plot(wavelength, lineflip,color='peachpuff')
   ax[0].set_xlim([300,800])
   ax[0].set_ylim(ymin=0)
   # ax[0].set_ylim([0, 20])
   ax[0].set_ylabel('Intensity', color=(0.7, 0.7, 0.7))
   ax[0].set_xlabel('Wavelength in nm', color=(0.7, 0.7, 0.7))
   for xmaj in ax[0].xaxis.get_majorticklocs():
       ax[0].axvline(x=xmaj, ls='-',color=(0.3, 0.3, 0.3),linewidth=0.5)
   for xmin in ax[0].xaxis.get_minorticklocs():
       ax[0].axvline(x=xmin, ls='--', color=(0.3, 0.3, 0.3), linewidth=0.5)
   for ymaj in ax[0].yaxis.get_majorticklocs():
       ax[0].axhline(y=ymaj, ls='-',color=(0.3, 0.3, 0.3),linewidth=0.5)
   for ymin in ax[0].yaxis.get_minorticklocs():
       ax[0].axhline(y=ymin, ls='--')

   ax[0].set_title('USB-CAM Spectrometer', color=(0.7, 0.7, 0.7))
   ax[0].tick_params(direction='inout', length=6, width=1.5, colors=(0.7, 0.7, 0.7),
                  grid_color=(0.7, 0.7, 0.7))
   plt.rcParams["figure.figsize"] = [6.4*2, 4.8*1.2]
   plt.tight_layout()
   # draw is called in the end, after calculating peak lines.
   plt.draw()

   # ******************** CAMERA PLOT in matplotlib ********************
   # frame[:,:,0], frame[:,:,2q]=frame[:,:,2], frame[:,:,0]
   ax[1].imshow(frame,cmap='gray',interpolation='bicubic')
   ax[1].set_title('USB-CAM LIVE FEED', color=(0.7, 0.7, 0.7))
   ax[1].tick_params(direction='inout', length=6, width=1.5, colors=(0.7, 0.7, 0.7),
                     grid_color=(0.7, 0.7, 0.7))
   plt.draw()

   # -------------------Fresh Plot from live-cam feed----


   plt.draw()
   plt.savefig('Graphs/LIBS coppper graph.png',facecolor=(0.2,0.2, 0.2,1.0))  # saving inside the for loop doesn't work
   print((time.time()-start)*1000,"ms")     # TIME for a loop
   # ------------------Saving the file-------------------
   if plt.waitforbuttonpress(0.01): # cv2.waitKey(1) and 0xFF == ord('q')
       data = np.array(np.arange(1, pix_width * 2 + 1,dtype=float).reshape(pix_width, 2))
       for i in range(pix_width):
           # print(np.shape(data))
           # print(wavqelength[-1])
           data[i, 0] = 1.0 * wavelength[i]
           data[i, 1] = 1.0 * lineflip[i]
       np.savetxt('Libs coppper data.csv', data ,fmt='%3.2f', delimiter=',') #graph data

       f = open('DATA/Libs coppper data.csv', 'a') # to append peak values append
       f.write('\n Peaks-')
       for i in range(c):
           f.write(" ")
           f.write(str(peak[i]))   # peak values
       f.close()
       plt.close()
       capture_frame.release()
       cv2.destroyAllWindows()
       break
   # ------------------Saving the file ends-------------------
   plt.pause(0.01)
capture_frame.release()
cv2.destroyAllWindows()