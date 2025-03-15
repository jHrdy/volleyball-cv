import numpy as np
import cv2 as cv
from random import randint

cap = cv.VideoCapture('../vb1.mp4')
if not cap.isOpened():
    print("Can not open video")
    exit()
i = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    if randint(0,1000)> 996:
        cv.imwrite(f'frames/{i}.jpg', frame)
        i += 1
     
print(f'num of frames {i}')
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()