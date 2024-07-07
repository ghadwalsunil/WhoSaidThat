import face_recognition
import os, sys
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/sunil/projects/Stuff/Combined/WhoSaidThat")

from who_said_that.utils.model.faceDetector import S3FD

DET = S3FD(device="cuda")

input_file = "/vol3/sunil/output/video_temp/zzz_PiersMorgan_1_165_368/pyframes/003197.jpg"

image = cv2.imread(input_file)

image_fr = face_recognition.load_image_file(input_file)

print(datetime.now())
face_locations = face_recognition.face_locations(image_fr)
print(datetime.now())
face_embeddings = face_recognition.face_encodings(image_fr, face_locations, model="small")
print(datetime.now())
print(len(face_locations))

# print(face_embeddings)

for face_location in face_locations:
	top, right, bottom, left = face_location
	print("Face found at top: {}, left: {}, bottom: {}, right: {}".format(top, left, bottom, right))

for face_location in face_locations:
	top, right, bottom, left = face_location
	cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

# cv2.imwrite("output_1.jpg", image)

imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("S3FD")
print(datetime.now())
bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])
print(datetime.now())

for bbox in bboxes:
	left, top, right, bottom, _ = bbox
	print("Face found at top: {}, left: {}, bottom: {}, right: {}".format(top, left, bottom, right))

for bbox in bboxes:
	left, top, right, bottom, _ = bbox
	print(type(bbox))
	# convert to int
	left, top, right, bottom = int(left), int(top), int(right), int(bottom)
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imwrite("output.jpg", image)

# Face found at top: 800.7559204101562, left: 417.61114501953125, bottom: 1035.8226318359375, right: 110.89877319335938
# Face found at top: 110.89877319335938, left: 800.7559204101562, bottom: 417.61114501953125, right: 1035.8226318359375
# Face found at top: 142, left: 765, bottom: 409, right: 1033