import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition as fr

img = cv2.imread("/home/levanlinh/Pictures/Opencv/drTruong.jpg", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgtest = cv2.imread("/home/levanlinh/Pictures/Opencv/drTruongtest.jpg", cv2.IMREAD_COLOR)
#imgtest = cv2.imread("/home/levanlinh/Pictures/Opencv/shark_Hung.jpg", cv2.IMREAD_COLOR)
imgtestRGB = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgRGB)[0]
encodeTruong = fr.face_encodings(imgRGB)[0]
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

faceLocTest = fr.face_locations(imgtestRGB)[0]
encodeTruongTest = fr.face_encodings(imgtestRGB)[0]
cv2.rectangle(imgtest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 0, 255), 2)

results = fr.compare_faces([encodeTruong], encodeTruongTest)
faceDis = fr.face_distance([encodeTruong], encodeTruongTest)
print(results, faceDis)
cv2.putText(imgtest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Dr Truong", img)
cv2.imshow("Dr Truong test", imgtest)
cv2.waitKey(0)
cv2.destroyAllWindows()