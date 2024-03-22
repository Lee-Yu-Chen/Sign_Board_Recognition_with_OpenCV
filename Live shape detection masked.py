import numpy as np
import cv2 as cv


vid=cv.VideoCapture(0)

while 1:
    frame=vid.read()[1]
    Frame=frame.copy()


    scale=0.5
    #img=cv.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)),cv.INTER_AREA)

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(gray,200,700,apertureSize=5)

    # 40,200,3 
    # 50,250,3
    # 200, 500, 3
    # the bigger the apertureSize(can only be 3/5/7), the more details it remain in the Canny image


    lines=cv.HoughLines(edges,1,np.pi/180,170)
    # the bigger the threshold value, the stricter it choose lines

    #print(lines)

    if lines is None:
        a=1
    else:
      

        yup=[]
        ydown=[]
        xleft=[]
        xright=[]
        
        for line in lines:
            rho,theta=line[0]
            a=np.cos(theta)
            b=np.sin(theta)
            x0=a*rho
            y0=b*rho

            x1=int(x0+1000*(-b))

            y1=int(y0+1000*(a))

            x2=int(x0-1000*(-b))

            y2=int(y0-1000*(a))

            cv.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
            #(x1,y1) and (x2,y2) are two points that the lines connected

            cv.circle(frame,(x0,y0),20,(255,0,0),-1)


            #print((x0,y0))

            if x0<10 and y0<(frame.shape[0]/2):
                yup.append(y0)
            elif x0<10 and y0>(frame.shape[0]/2):
                ydown.append(y0)
            elif y0<10 and x0<(frame.shape[1]/2):
                xleft.append(x0)
            elif y0<10 and x0>(frame.shape[1]/2):
                xright.append(x0)
            else:
                yup=yup
                


            

        yup.sort()
        ydown.sort()
        xleft.sort()
        xright.sort()
        
        if len(yup)!=0 and len(ydown)!=0 and len(xleft)!=0 and len(xright)!=0:

            blank=np.zeros(frame.shape[:2],dtype='uint8')
            
            cv.rectangle(blank,(xleft[-1],yup[-1]),(xright[0],ydown[0]),(255,255,255),-1)

            masked=cv.bitwise_and(Frame,Frame,mask=blank)

            cv.imshow('Masked',masked)
            

            
            ###
            graymasked=cv.bitwise_and(gray,gray,mask=blank)
            
            _,thrash=cv.threshold(graymasked,130,255,cv.THRESH_BINARY)

            # threshold = 130 if open cam, 240 if open pic

            img=masked

            cv.imshow('threshold',thrash)

            contours, _ =cv.findContours(thrash,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

            # method = cv.CHAIN_APPROX_SIMPLE / cv.CHAIN_APPROX_NONE

            for contour in contours :

                area = cv.contourArea(contour)

                if area>400:
                    
                
                
                    #approx=cv.approxPolyDP(contour, 0.01*cv.arcLength(contour,1),1)
                    approx=cv.approxPolyDP(contour, 3,1)
                    # epsilon=3
    
                    cv.drawContours(img,[approx],0,(0,0,255), 2)
                    cv.drawContours(thrash,[approx],0,(0,0,255), 2)
                    x=approx.ravel()[0]
                    y=approx.ravel()[1]

                    if len(approx)==3:
                        cv.putText(img,"Triangle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
        
                    elif len(approx)==4:
                        x,y,w,h=cv.boundingRect(approx)
                        aspectRatio=float(w)/h
                        #print(aspectRatio)
                        if aspectRatio>=0.95 and aspectRatio<=1.05:
                            cv.putText(img,"Square",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                        else:
                            #cv.putText(img,"Rectangle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                            gg=0

                    else:
                        cv.putText(img,"Circle",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                    '''
                    elif len(approx)==5:
                        cv.putText(img,"Pentagon",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

                    elif len(approx)==10:
                        cv.putText(img,"Star",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))
                        '''
                
                

                    cv.drawContours(img,contour,0,(0,255,0), 5)
        
            #cv.imshow('threshold',thrash)
            cv.imshow("Shapes Detection",img)
            

            
        
        
            
        
                    

            
    
    cv.imshow('lines',frame)
    #cv.imshow('Canny',edges)
    cv.waitKey(1)
    

