import numpy as np
import cv2 as cv

filename = 'C:/Users/ACER_USER/Downloads/Symbols/Picture1.png'
img = cv.imread(filename)
blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

scale = 0.5
resized=cv.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation=cv.INTER_AREA)

cv.imshow('dst',resized)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
