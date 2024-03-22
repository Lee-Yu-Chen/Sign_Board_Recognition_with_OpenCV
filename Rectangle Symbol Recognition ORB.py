#https://www.youtube.com/watch?v=gbL3XKOiBvw


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

vid=cv.VideoCapture(0)

traffic=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Modified/Pic5.png') 
shapes=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Modified/Pic7.png')
face=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Modified/Pic8.png')
measuredis=cv.imread('C:/Users/ACER_USER/Documents/Academic/Project Week/Sem 2/Symbols/Modified/Pic9.png')


scale=0.5
traffic=cv.resize(traffic,(int(traffic.shape[1]*scale),int(traffic.shape[0]*scale)),cv.INTER_AREA)
shapes=cv.resize(shapes,(int(shapes.shape[1]*scale),int(shapes.shape[0]*scale)),cv.INTER_AREA)
face=cv.resize(face,(int(face.shape[1]*scale),int(face.shape[0]*scale)),cv.INTER_AREA)
measuredis=cv.resize(measuredis,(int(measuredis.shape[1]*scale),int(measuredis.shape[0]*scale)),cv.INTER_AREA)


Traffic=traffic.copy()
Shapes=shapes.copy()
Face=face.copy()
Measuredis=measuredis.copy()


Traffic=cv.cvtColor(Traffic,cv.COLOR_BGR2GRAY)
Shapes=cv.cvtColor(Shapes,cv.COLOR_BGR2GRAY)
Face=cv.cvtColor(Face,cv.COLOR_BGR2GRAY)
Measuredis=cv.cvtColor(Measuredis,cv.COLOR_BGR2GRAY)

'''
cv.imshow("1",Traffic)
cv.imshow("2",Shapes)
cv.imshow("3",Face)
cv.imshow("4",Measuredis)
'''



while 1:
    frame=vid.read()[1]
    Frame=frame.copy()


    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(gray,40,200,apertureSize=3)
    
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

            maskedgray=cv.cvtColor(masked,cv.COLOR_BGR2GRAY)

            orb=cv.ORB_create(nfeatures=2000,WTA_K=4)

            kp1, des1 = orb.detectAndCompute(maskedgray,None)
            kp2, des2 = orb.detectAndCompute(Traffic,None)
            kp3, des3 = orb.detectAndCompute(Shapes,None)
            kp4, des4 = orb.detectAndCompute(Face,None)
            kp5, des5 = orb.detectAndCompute(Measuredis,None)

            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            if des1 is None:
                gg=0
            else:
                li=["Traffic","Shapes","Face","Measuredis"]
                li1=[des2,des3,des4,des5]
                
                for i in range(len(li1)):
                    matches=bf.match(des1,li1[i])
                    matches=sorted(matches,key=lambda x:x.distance)
                    nice=0
                    total=0

                    for j in matches:
                        total+=j.distance

                        if j.distance<=30:
                            nice+=1
                        else:
                            nice=nice


                    print(li[i]," : ",len(matches),' ',nice,' ',total,' ',total/len(matches))

                print('\n')


                                  
                
            
        
        
            
        
                    

            
    
    cv.imshow('lines',frame)
    #cv.imshow('Canny',edges)
    cv.waitKey(1)
    

