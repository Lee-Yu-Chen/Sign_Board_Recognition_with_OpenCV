import cv2 
import numpy as np
#import go
import time
#import RPi.GPIO as gp
from skimage.metrics import structural_similarity as ssim


'''
gp.setmode(gp.BCM)

# we are using BCM
led=17
in1=27
in2=22
in3=23
in4=24
ena=5
enb=6

gp.setwarnings(False)

gp.setup(led,gp.OUT)

gp.setup(in1,gp.OUT)
gp.setup(in2,gp.OUT)
gp.setup(in3,gp.OUT)
gp.setup(in4,gp.OUT)

gp.setup(ena,gp.OUT)
gp.setup(enb,gp.OUT)


pwm1=gp.PWM(ena,100) # 100 is the frequency
pwm2=gp.PWM(enb,100)
pwm1.start(30)        # 30 is the duty cycle/high time in a periodic pulse
pwm2.start(30)        # in this situation, this is the analogue voltage value supplied to the motor(the speed)


'''



cap = cv2.VideoCapture(0)
scale = 3
width = 248 * scale
height = 190 * scale
scaleofdisplay = 0.75

img2 = cv2.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Picture5.png') 
img2 = cv2.resize(img2, (664, 490),cv2.INTER_AREA)

img1 = cv2.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Picture9.png') 
img1 = cv2.resize(img1, (664, 490),cv2.INTER_AREA)
#cv2.imshow('img1',img1)

def getContours(img, MinArea = 1000, filter = 0,draw = False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    kernel = np.ones((5, 5))
    
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=2)
    
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finalContours = []
    
    for i in contours:
        
        area = cv2.contourArea(i)
        
        if area > MinArea:
            
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            
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
    NewPoints = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    
    add = myPoints.sum(1)
    NewPoints[0] = myPoints[np.argmin(add)]
    NewPoints[3] = myPoints[np.argmax(add)]
    
    diff = np.diff(myPoints, axis=1)
    NewPoints[1] = myPoints[np.argmin(diff)]
    NewPoints[2] = myPoints[np.argmax(diff)]
    
    return NewPoints


def warpImg (img, points, w, h, pad=40):
    coor = [0, 0], [w, 0], [0, h], [w, h]
    points = reorder(points)
    
    pts1 = np.float32(points)
    pts2 = np.float32(coor)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    
    return imgWarp


def mse(img1, img):

    err = np.sum((img1.astype("float") - img.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    return err


def compare_images(img1, img):

    m = mse(img1, img)
    s = ssim(img1, img)

    return s


symbol_1 = 1
symbol_2 = 1
forward = False
speed = 0.25
move_time = 0
distance = 0
first_lap = 0
second_lap = 0

s1=1
print('ok')
while s1>0.35:
    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_180)

    img, finalContours = getContours(img, MinArea=50000, filter=4)

    cv2.imshow('',img)
    cv2.waitKey(0)
    
    if len(finalContours) != 0:
        biggest = finalContours[0][2]
        # print(biggest)
        imgWarp = warpImg(img, biggest, width, height)

        cv2.imshow('ROI', imgWarp)

        tem_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        warp_gray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)

        s1 = compare_images(tem_gray, warp_gray)
    else:
        print()
    print('s1 : ',s1)

        
first_lap = time.time()
'''
gp.output(in1,gp.HIGH)
gp.output(in2,gp.LOW)
gp.output(in3,gp.HIGH)
gp.output(in4,gp.LOW)
'''



s2=0
    
while s2<0.35:
    success, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_180)

    img, finalContours = getContours(img, MinArea=50000, filter=4)
    
    if len(finalContours) != 0:
        biggest = finalContours[0][2]
        # print(biggest)
        imgWarp = warpImg(img, biggest, width, height)

        cv2.imshow('ROI', imgWarp)

        tem_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        warp_gray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)

        s2 = compare_images(tem_gray, warp_gray)
    

        #print(s)
        time.sleep(1)
        '''        
        if s > 0.35 and symbol_1 == 1:

            symbol_1 = 0
            forward = True
            first_lap = time.time()

        elif s < 0.35 and symbol_2 == 1:

            symbol_2 = 0
            forward = False
            second_lap = time.time()
            #print('STOP')
            go.stop()

    if forward:
        #print('FORWARD')
        go.forward()
    '''
    print('s2 : ',s2)
    
    '''
gp.output(in1,gp.LOW)
gp.output(in2,gp.LOW)
gp.output(in3,gp.LOW)
gp.output(in4,gp.LOW)
'''
#time.sleep(1)
second_lap=time.time()

move_time = second_lap - first_lap
        
distance = move_time * speed

if distance > 0:

    print(distance, ' m')


    '''
    
    img = cv2.resize(img, (0, 0), None, scaleofdisplay, scaleofdisplay)

    cv2.imshow('CameraFeed', img)
    # use imgWarp as the Region of Interest
    k = cv2.waitKey(1)
    if k == 27:
        break
    '''


cv2.destroyAllWindows()
