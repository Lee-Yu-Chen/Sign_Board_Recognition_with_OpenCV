import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
img1 = cv2.imread('/home/pi/Desktop/New folder (3)/Picture6.png') #STOP template
img1 = cv2.resize(img1, (702,540))

while True:
    success, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    orb = cv2.ORB_create()

    #DetectCircles
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, frame.shape[1]*2, param1=200, param2=50, minRadius=int(blur.shape[1]/8), maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        mask = np.full((frame.shape[0],frame.shape[1]), 0, dtype=np.uint8)
        
        for i in circles[0,:]:
            #draw outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
            #draw center of circle
            cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)
            #draw masked circle
            cv2.circle(mask, (i[0],i[1]),i[2],(255,255,255),-1)

        fg = cv2.bitwise_or(frame,frame,mask=mask)

        mask = cv2.bitwise_not(mask)
        background = np.full(frame.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(background,background,mask=mask)

        final = cv2.bitwise_or(fg,bk)
        cv2.imshow('hmm', final)

        #Feature matching
        kp1,des1 = orb.detectAndCompute(img1,None)
        kp2,des2 = orb.detectAndCompute(final,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2) #Comparing STOP template with masked frame

        matches = sorted(matches, key = lambda x:x.distance)

        if len(matches) > 170:
            print("STOP")

        else:
            print("Other Symbols")

        #print(len(matches))

        #img3 = cv2.drawMatches(img1,kp1,final,kp2,matches[:50],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #plt.imshow(img3),plt.show()

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xFF #Terminate while press ESC
    if k == 27:
        break


cv2.destroyAllWindows()