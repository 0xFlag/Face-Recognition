#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import face_recognition
import cv2

# Open the input movie file
input_movie = cv2.VideoCapture("hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Open the input image file
image = face_recognition.load_image_file("img.jpg")
images = face_recognition.face_encodings(image)[0]

frame_number = 0

while True:
	# Grab a single frame of video
	ret, frame = input_movie.read()
	frame_number += 1

	# Quit when the input video file ends
	if not ret:
		break

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_frame = frame[:, :, ::-1]

	# Find all the faces and face encodings in the current frame of video
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

	print("Writing frame {} / {}".format(frame_number, length))

	matche = face_recognition.compare_faces(images, face_encodings)
	
	#face_distances = face_recognition.face_distance(images, face_encodings)
	#for i, face_distance in enumerate(face_distances):

	for i in matche:
		if(i == True):
			print("True")
			cv2.imwrite("./True/" + str(frame_number) + ".jpg",frame)
		elif(i == False):
			print("False")
			cv2.imwrite("./False/" + str(frame_number) + ".jpg",frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()