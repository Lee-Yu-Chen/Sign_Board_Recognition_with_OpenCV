import cv2 as cv

a=cv.VideoCapture(0)



for i in range(500):
    frame=a.read()[1]
    play=a.read()[0]
    cv.imshow("Test video",frame)
    cv.waitKey(1)
    


a.release()
cv.destroyAllWindows()
