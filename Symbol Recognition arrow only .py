# the steps to detect symbol :
# 1. circle detection with HoughCircles()
# 2. masking with bitwise_and
# 3. corner detecting with goodFeaturesToTrack(), use only 2 corners
# 4. calculate the corners that is above, below, right to and left to the centre of circle detected
# 5. if more dots (corners) are located above/below/left/right to the centre, then the symbols are forward/reverse/left/right


import cv2 as cv
import numpy as np


vid= cv.VideoCapture(0)

while 1:
    frame=vid.read()[1]

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.medianBlur(gray, 5)


    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, frame.shape[1]*2, param1=200, param2=50, minRadius=int(frame.shape[1]/8), maxRadius=0)
    # param2 is directly related to how strict the detector will detect, the higher the stricter, but the lower the lagger(more to compute)

    #print(circles)
    
    
    blank=np.zeros(frame.shape[:2],dtype='uint8')
    mask=blank.copy()
    mask2=blank.copy()
    
    masked=blank
    Frame=frame.copy()

    
    x0=0
    y0=0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        #print(circles)
        #print('\n')

        
        i=circles[0][0]
        # Draw outer circle
        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        

        cv.circle(mask, (i[0], i[1]), i[2]-10,(255, 255,255), -1)
        cv.circle(mask2,(i[0], i[1]), i[2]-10,(255, 255,255), -1)
        
        masked=cv.bitwise_and(Frame,Frame,mask=mask)

        x0=circles[0][0][0]
        y0=circles[0][0][1]
        
    
    cv.imshow('circle',frame)
    cv.imshow('Masked',masked)
    #cv.imshow('Mask',mask)
    Masked=masked.copy()
    

    
    
    #gray1 = cv.cvtColor(masked,cv.COLOR_BGR2GRAY)
    gray1=cv.bitwise_and(gray,gray,mask=mask2)
    corners = cv.goodFeaturesToTrack(gray1,6,0.1,25)
    # 9,0.05,30
    # 2nd param = maximum number of corners, len(corners)
    # 3rd param = how strict the function find corner
    # 4th param = minimun distance between each corner
   

    
    
    if corners is None:
        a=1
    else:
        
        
        corners = np.int0(corners)

        # x posotive, x negative, y positive, y negative
        Xp=0
        Xn=0
        Yp=0
        Yn=0
       
        for i in corners:
            x,y = i.ravel()
            cv.circle(Masked,(x,y),5,(0,0,255),-1) # 3rd param = radius of dots


            # note that in pixel coordinating system, verticle downward is y+ve, verticle upward is y-ve
            # origin (0,0) is located at the top left corner of the picture
            if x>x0 :
                Xp=Xp+1
                
            elif x<x0 :
                Xn=Xn+1
                
            if y>y0 :
                Yn=Yn+1
                # Yn and Yp is based in normal cartesian coordinating system

            elif y<y0 :
                Yp=Yp+1
            
        
        cv.imshow('corner',Masked)

        n=len(corners)

        '''
        if Yp-Yn>=4 and Xp-Xn<2 :
            print('Forward\n')
            
        elif Yn-Yp>=4 and Xp-Xn<2 :
            print('Reverse\n')
            
        elif Xp-Xn>=4 and Yp-Yn<2 :
            print('Right\n')
            
        elif Xn-Xp>=4 and Yp-Yn<2 :
            print('Left\n')

        else :
            print('Undefined\n')
        '''

        
        if Yp>Yn and (Xp-Xn)**2<2 :
            #Forward
            print('Forward\n')
            
        elif Yp<Yn and (Xp-Xn)**2<2 :
            #Reverse
            print('Reverse\n')
            
        elif Xp>Xn and (Yp-Yn)**2<2 :
            print('Right\n')
            
        elif Xp<Xn and (Yp-Yn)**2<2 :
            print('Left\n')

        else :
            print('Undefined\n')



    cv.waitKey(1)

vid.release()
cv.destroyAllWindows()

