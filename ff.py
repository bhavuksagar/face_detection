import cv2
import numpy as ns

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)

while True:
    check,frame=cam.read()
    faces=face.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,220),2)
    cv2.imshow("face",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindow()
