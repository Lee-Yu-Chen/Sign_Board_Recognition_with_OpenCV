import cv2
import numpy as np
import time

# ###Settings### #
Webcam = True
image = 'Picture4.png'
cap = cv2.VideoCapture(0)
cap.set(10, 160)  # Brightness
cap.set(3, 1920)  # Width
cap.set(4, 1080)  # Height
scale = 3
width = 248* scale
height = 190* scale
scaleofdisplay = 0.75


def millis():
    return round(time.time()*1000)


def getContours(img, cThr=[100, 100], showCanny=False, MinArea = 1000, filter = 0,draw = False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny: cv2.imshow('Canny', imgThreshold)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > MinArea:
            perimeter = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*perimeter,True)
            bbox = cv2.boundingRect(approx)
            # can filter and say that we only want rectangle
            b = [len(approx), area, approx, bbox, i]
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append(b)
            else:
                finalContours.append(b)
    finalContours = sorted(finalContours,key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 5)
    return img, finalContours


def reorder(myPoints):
    print(myPoints.shape)
    NewPoints = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    NewPoints[0] = myPoints[np.argmin(add)]
    NewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    NewPoints[1] = myPoints[np.argmin(diff)]
    NewPoints[2] = myPoints[np.argmax(diff)]
    print('reordered')
    return NewPoints


def warpImg (img, points, w, h, pad=40):
    hey = [0, 0], [w, 0], [0, h], [w, h]
    points = reorder(points)
    print(points)
    pts1 = np.float32(points)
    pts2 = np.float32(hey)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp


while True:
    flagROI = False
    if Webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(image)
    time_before = millis()
    img, finalContours = getContours(img, MinArea=50000, filter=4)
    if len(finalContours) != 0:
        flagROI = True
        print('printing image')
        biggest = finalContours[0][2]
        # print(biggest)
        imgWarp = warpImg(img, biggest, width, height)
        cv2.imshow('ROI', imgWarp)
        t = 0
        print('Have ROI')
    else:
        time_now = millis()
        # print('Time now = ' + str(time_now))
        # print('Time before = ' + str(time_before))
        t = time_now - time_before
        print('No ROI')
    print('Time interval = ' + str(t))
    if t > 9:
        flagROI = False
        # window destroyed, or rectangle npt detected
        cv2.destroyWindow('ROI')
        print('ByeBye')
    print('flagROI = ' + str(flagROI))
    img = cv2.resize(img, (0, 0), None, scaleofdisplay, scaleofdisplay)
    cv2.imshow('CameraFeed', img)
    # use imgWarp as the Region of Interest
    cv2.waitKey(1)
