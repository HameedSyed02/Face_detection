# import module
import cv2

# Read the haarcascade file (trained data)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose a image to detect
img = cv2.imread('photo.jpg')

# convert BGR image to gray image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (128, 144), (128+296, 144+296), (0, 0, 255), 2)

# show image
cv2.imshow('facedetect_hameed', img)

# wait key holds the window
cv2.waitKey()

print('program completed')