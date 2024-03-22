#https://learnopencv.com/opencv-qr-code-scanner-c-and-python/

import cv2 as cv
import numpy as np

qr=cv.QRCodeDetector()


img=cv.imread("C:/Users/ACER_USER/Desktop/1.jpg")

msg,box,rectI=qr.detectAndDecode(img)

print(msg,'\n\n')
print(box)

if len(msg)>0:
    
    #rectT=np.uint8(rectI)
    cv.imshow("Rectified QR Code",rectI)


   
    start=(int(box[0][0][0])-10,int(box[0][0][1])-10)
    end=(int(box[0][2][0])+10,int(box[0][2][1])+10)
    cv.rectangle(img,start,end,(255,0,0),1)

    cv.imshow('asd',img)

else:
    print('QR Code not found')


cv.waitKey(0)
cv.destroyAllWindows()
    
