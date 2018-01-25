import cv2
import numpy as np

cap = cv2.VideoCapture('video-9.avi')

flag, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100,apertureSize = 3)

lines = cv2.HoughLinesP(edges, 2, np.pi/180, 200, minLineLength = 150, maxLineGap = 20)

while(cap.isOpened()):
    flag, img = cap.read()
    if flag == True:
        
        for line in lines:
        #for i in range(0, len(lines)):
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0,0,255], 3)

        cv2.imshow('Video',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
