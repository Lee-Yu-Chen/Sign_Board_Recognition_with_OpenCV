import cv2 as cv

a=cv.VideoCapture("C:/Users/ACER_USER/Downloads/InS Tutorial 18 March .mp4")


play=1
for i in range(500):
    frame=a.read()[1]
    play=a.read()[0]
    cv.imshow("Test video",frame)
    cv.waitKey(1)
    print(play)


a.release()
cv.destroyAllWindows()
