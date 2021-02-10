# import moudules
from random import randrange

import cv2

# read the trained rata
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture image
webcam = cv2.VideoCapture(0)

#looping hte each frame
while True:
    success_data_read, frames =  webcam.read() # read each frame from webcame
    grayscale_capture = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) # GRAY FRAME CONVERTING
    face_coordinates = trained_data.detectMultiScale(grayscale_capture) # locating real time coordinates
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)
    cv2.imshow('FACE_DETECTION_REALTIME_HAMEED', frames)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break


print('program complted')