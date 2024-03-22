import cv2 as cv
import numpy as np

vid= cv.VideoCapture(0)

while 1:
    play,frame=vid.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.medianBlur(gray, 5)


    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, frame.shape[1]*2, param1=200, param2=50, minRadius=int(frame.shape[1]/8), maxRadius=0)
    # param2 is directly related to how strict the detector will detect, the higher the stricter, but the lower the lagger(more to compute)

    print(circles)
    
    
    blank=np.zeros(frame.shape[:2],dtype='uint8')
    masked=blank
    Frame=frame.copy()

    
     
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles)
        print('\n')

        
        i=circles[0][0]
        # Draw outer circle
        cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        

        cv.circle(blank, (i[0], i[1]), i[2],(255, 255,255), -1)
        
        masked=cv.bitwise_and(Frame,Frame,mask=blank)
        
    
    cv.imshow('circle',frame)
    cv.imshow('Masked',masked)
    cv.imshow('Mask',blank)
    
    cv.waitKey(1)

vid.release()
cv.destroyAllWindows()

