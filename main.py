import cv2
import os

img_path = os.path.join('.', 'assets', 'basketball_team2_ar.jpg')

# load haar cascade face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectar caras
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20))
for (x, y, w, h) in faces:
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()