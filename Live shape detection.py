import numpy as np
import cv2 as cv


vid=cv.VideoCapture(0)

while 1:
    frame=vid.read()[1]
    Frame=frame.copy()


    scale=0.5
    #img=cv.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),cv.INTER_AREA)

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(gray,200,700,apertureSize=5)

    # 40,200,3 
    # 50,250,3
    # 200, 500, 3
    # the bigger the apertureSize(can only be 3/5/7), the more details it remain in the Canny image


   
            
    _,thrash=cv.threshold(edges,130,255,cv.THRESH_BINARY)

    # threshold = 130 if open cam, 240 if open pic

    img=frame

    cv.imshow('threshold',thrash)

    contours, _ =cv.findContours(thrash,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # method = cv.CHAIN_APPROX_SIMPLE / cv.CHAIN_APPROX_NONE

    for contour in contours :
        

        area = cv.contourArea(contour)

        if area>400:
                    
                
                
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
                #print(aspectRatio)
                if aspectRatio>=0.95 and aspectRatio<=1.05:
                    cv.putText(img,"Square",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                else:
                    #cv.putText(img,"Rectangle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                    gg=0

            else:
                cv.putText(img,"Circle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
            '''
            elif len(approx)==5:
                cv.putText(img,"Pentagon",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

            elif len(approx)==10:
                cv.putText(img,"Star",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
            '''
                
                

            cv.drawContours(img,contour,0,(0,255,0), 5)
        
    #cv.imshow('threshold',thrash)
    cv.imshow("Shapes Detection",img)
            

            
    
    #cv.imshow('lines',frame)
    #cv.imshow('Canny',edges)
    cv.waitKey(1)
    

