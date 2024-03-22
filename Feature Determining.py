# https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/ACER_USER/Downloads/Symbols/Picture6.png',0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()
cv2.waitKey(0)
print(img2)
