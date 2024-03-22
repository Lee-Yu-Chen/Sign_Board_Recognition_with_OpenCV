# https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/
# This code is extremely laggy ran on raspberry pi
# The workload in the for loop is too heavy

import cv2 as cv

a=cv.VideoCapture(0)

scale=1.2



for i in range(300):
    frame=a.read()[1]
    play=a.read()[0]

    
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)

    resizedframe=cv.resize(frame,dimension,cv.INTER_AREA)
    

    rotateframe=cv.rotate(resizedframe,cv.ROTATE_180)

    
    #cv.imshow("Test video",frame)
    cv.imshow("Rotated",rotateframe)

    
    cv.waitKey(1)
    print(play)


a.release()
cv.destroyAllWindows()
