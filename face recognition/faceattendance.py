import cv2
import numpy as np 
import face_recognition as fr
import os
from datetime import datetime

def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = fr.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList 

def markAttendance(name):
	with open('Attendance.csv', 'r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtString = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name}, {dtString}')

path = 'imageAttendance'
images = []
labels = []
myList = os.listdir(path)
print(myList)
for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	labels.append(os.path.splitext(cl)[0]) #chia chuoi thanh ten  labels

print(labels)

encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)#("face.mp4")
name_com = "sfd"

while True:
	_, frame = cap.read()
	frameS = cv2.resize(frame, None, fx = 0.25, fy = 0.25)
	frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

	facesCurFrame = fr.face_locations(frameS)
	encodeCurFrame = fr.face_encodings(frameS, facesCurFrame)

	for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
		#matches = fr.compare_faces(encodeListKnown, encodeFace)
		faceDis = fr.face_distance(encodeListKnown, encodeFace)
		if(faceDis[0] < 0.5):
			matchIndex = np.argmin(faceDis)
			#print(matchIndex)
			name = labels[matchIndex]
		else:
			name = "Unknown"

		#print(name)
		y1, x2, y2, x1 = faceLoc
		y1, x2, y2, x1 = y1*4-35, x2*4+15, y2*4+10, x1*4-15
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.rectangle(frame, (x1, y1-15), (x1+90, y1), (0, 255, 0), -1)
		cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
		markAttendance(name)

		#If unknown => write
		if(name != name_com and name == "Unknown"):
			name_com = "Unknown"
			roi = frame[y1:y2, x1:x2]
			cv2.imwrite("Unkown.jpg", roi)

	cv2.imshow("Video", frame)
	k = cv2.waitKey(2)&0xFF
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()