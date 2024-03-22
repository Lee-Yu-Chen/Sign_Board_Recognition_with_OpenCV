#https://www.youtube.com/watch?v=oXlwWbU8l2o
#12:57 - 17:25
import cv2 as cv

a=cv.VideoCapture("/home/pi/Desktop/a.mp4")

scale=0.75

play=1
while play:
    frame=a.read()[1]
    play=a.read()[0]

    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    
    newframe=cv.resize(frame,dimension,cv.INTER_AREA)
    
    cv.imshow("Test video",frame)
    cv.imshow("Resized",newframe)

    
    cv.waitKey(1)
    print(play)


a.release()
cv.destroyAllWindows()
