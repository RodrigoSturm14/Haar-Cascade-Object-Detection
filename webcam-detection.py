import cv2
import os

video_path = cv2.VideoCapture(0)
# load haar cascade face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ret = True
while ret:
  ret, frame = video_path.read()
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # detectar caras
  faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=6, minSize=(20, 20))
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

  cv2.imshow('img', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_path.release()
cv2.destroyAllWindows()