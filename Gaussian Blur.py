import cv2
import numpy
  

src = cv2.imread('C:/Users/ACER_USER/Downloads/Profile Picture/1.jpg')
 

blurred= cv2.GaussianBlur(src,(13,13),cv2.BORDER_DEFAULT)
# kernel size mean the degree of blurring
# kernel size can only be odd number 

cv2.imshow("Blurred",blurred)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
