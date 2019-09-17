#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import face_recognition
import os
import shutil

dir="images"
files=os.listdir(dir)
for file in files:
    filepath=dir+"/"+file
    image = face_recognition.load_image_file(filepath)

    biden_encoding = face_recognition.face_encodings(image)[0]

    unknown_image = face_recognition.load_image_file("./img.jpg")

    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

    # print("Is the unknown face a new person that we've never seen before? {}".format(results))
    
    face_distances = face_recognition.face_distance([biden_encoding], unknown_encoding)

    for i, face_distance in enumerate(face_distances):
    	print("The test image has a distance of {:.2f}% from known image.".format(face_distance * 100))
    	print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    	print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
    	
    imgsplit = os.path.abspath(filepath)
    for i in results:
    	if(i == True):
    		print(filepath + "\000True\r\n")
    		shutil.copy(filepath,"./True/")
    	elif(i == False):
    		print(filepath + "\000False\r\n")
    		#shutil.copy(filepath,"./False/")