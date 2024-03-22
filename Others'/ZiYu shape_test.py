import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def getContours(frame, canny):

    triangle = 0
    square = 0
    circle = 0
    
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        #print(area)

        if area > 500:
            cv2.drawContours(frame, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = 'Triangle'
                triangle += 1

            elif objCor == 4:
                objectType = 'Square'
                square += 1

            else:
                objectType = 'Circle'
                circle += 1

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, objectType,(x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)

    print('Total shapes detected: ', triangle + square + circle)
    print('Triangle: ', triangle)
    print('Square: ', square)
    print('Circle: ', circle)
    print('\n')



while True:
    success, frame = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),1)
    canny = cv2.Canny(blur,50,50)

    getContours(frame, canny)

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()
