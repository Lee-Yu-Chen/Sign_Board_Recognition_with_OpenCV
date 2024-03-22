#https://www.youtube.com/watch?v=gbL3XKOiBvw


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Picture8.png')

scale=0.5
img=cv.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),cv.INTER_AREA)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

edges=cv.Canny(gray,50,150,apertureSize=3)

lines=cv.HoughLines(edges,1,np.pi/180,200)

print(lines)

for line in lines:
    rho,theta=line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho

    x1=int(x0+1000*(-b))
    # x1 stores the rounded off value of r*cos(theta)-1000*sin(theta)

    y1=int(y0+1000*(a))
    # y1 stores the rounded off value of r*sin(theta)+1000*cos(theta)

    x2=int(x0-1000*(-b))
    # x2 stores the rounded off value of r*cos(theta)+1000*sin(theta)

    y2=int(y0-1000*(a))
    # y2 stores the rounded off value of r*sin(theta)-1000*cos(theta)

    cv.line(img,(x1,y1),(x2,y2),(255,255,255),2)
    #(x1,y1) and (x2,y2) are two points that the lines connected

cv.imshow('lines',img)
cv.waitKey(0)
cv.destroyAllWindows()
