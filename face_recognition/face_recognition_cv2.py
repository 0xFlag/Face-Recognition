#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import face_recognition
import cv2
import os

# Load the jpg file into a numpy array
dir="images"
files=os.listdir(dir)
filepath=dir+"/img.jpg"

'''
dir="images"
files=os.listdir(dir)
for file in files:
	filepath=dir+"/"+file
'''

image = cv2.imread(filepath)
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
	top, right, bottom, left = face_location
	print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

	imgsplit = os.path.split(filepath)[1]
	imgtitle = os.path.splitext(imgsplit)[0]

	face_image = image[top:bottom, left:right]
	cv2.imwrite("./faceimg/" + imgtitle + "_faceimg.jpg",face_image)

	cv2.rectangle(image, (left, top), (right, bottom), ( 0, 255,0), 2)
	cv2.imwrite("./testimg/" + imgtitle + "_testimg.jpg",image)

	cv2.imshow("Find Faces!", image)
	cv2.waitKey()