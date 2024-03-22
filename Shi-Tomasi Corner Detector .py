import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# cv use BGR
# matplotlib use RGB

img = cv.imread('C:/Users/ACER_USER/Downloads/Symbols/Picture1.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
# 2nd param = maximum number of corners
# 4nd param = minimun distance between each corner
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),10,255,-1) #10 = radius of circle
plt.imshow(img),plt.show()
