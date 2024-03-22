
import cv2 as cv

scale=0.5


a=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Picture5.png')


# img.shape return (height(y), width(x), 3/1)
width=int(a.shape[1]*scale)   # x value
height=int(a.shape[0]*scale)  # y vlaue
dimension=(width,height)
    
newa=cv.resize(a,dimension,cv.INTER_AREA)
    
#cv.imshow("Test",a)
cv.imshow("Resized",newa)

cv.waitKey(0)




    
    




