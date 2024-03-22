import numpy as np
import cv2 as cv

Img=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Picture7a.png')


scale=0.3
Img=cv.resize(Img,(int(Img.shape[1]*scale),int(Img.shape[0]*scale)),cv.INTER_AREA)

img=Img.copy()

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)


_,thrash=cv.threshold(gray,240,255,cv.THRESH_BINARY)


cv.imshow('threshold',thrash)

contours, _ =cv.findContours(thrash,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)







for contour in contours :
    #cv.drawContours(img,contour,0,(0,0,255), 5)
    #cv.drawContours(thrash,contour,0,(0,0,255), 5)

    #approx=cv.approxPolyDP(contour, 0.01*cv.arcLength(contour,1),1)
    approx=cv.approxPolyDP(contour, 3,1)
    # epsilon=3
    
    cv.drawContours(img,[approx],0,(0,0,255), 2)
    cv.drawContours(thrash,[approx],0,(0,0,255), 2)
    x=approx.ravel()[0]
    y=approx.ravel()[1]

    if len(approx)==3:
        cv.putText(img,"Triangle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
        
    elif len(approx)==4:
        x,y,w,h=cv.boundingRect(approx)
        aspectRatio=float(w)/h
        print(aspectRatio)
        if aspectRatio>=0.95 and aspectRatio<=1.05:
            cv.putText(img,"Square",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
        else:
            cv.putText(img,"Rectangle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
        
    elif len(approx)==5:
        cv.putText(img,"Pentagon",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

    elif len(approx)==10:
        cv.putText(img,"Star",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

    else:
        cv.putText(img,"Circle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

    cv.drawContours(img,contour,0,(0,255,0), 5)
        
cv.imshow('threshold',thrash)
cv.imshow("Shapes Detection",img)
cv.imshow('original',Img)
cv.waitKey(0)
cv.destroyAllWindows()
