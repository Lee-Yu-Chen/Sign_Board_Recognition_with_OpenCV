#https://learnopencv.com/opencv-qr-code-scanner-c-and-python/

import cv2 as cv


qr=cv.QRCodeDetector()


vid=cv.VideoCapture(0)
for i in range(500000):
    frame=vid.read()[1]

    msg,box,rectI=qr.detectAndDecode(frame)

    if len(msg)==0 :
        print('QR Code not found')

    else:
        print(msg)
        start=(int(box[0][0][0])-10,int(box[0][0][1])-10)
        end=(int(box[0][2][0])+10,int(box[0][2][1])+10)
            
        cv.rectangle(frame,start,end,(255,0,0),5)



        
    cv.imshow("Show the QR code",frame)
    cv.waitKey(1)
vid.release()
cv.destroyAllWindows()





