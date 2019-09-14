#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

from PIL import Image,ImageDraw
import face_recognition
import os

# Load the jpg file into a numpy array
dir="images"
files=os.listdir(dir)
filepath=dir+"/1.jpg"
image = face_recognition.load_image_file(filepath)

'''
dir="images"
files=os.listdir(dir)
for file in files:
    filepath=dir+"/"+file
    image = face_recognition.load_image_file(filepath)
'''

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    imgsplit = os.path.split(filepath)[1]
    imgtitle = os.path.splitext(imgsplit)[0]

    # You can access the actual face itself like this:
    # save face image
    face_image = image[top:bottom, left:right]
    pil_face_image = Image.fromarray(face_image)
    pil_face_image.save('./faceimg/' + imgtitle + '_faceimg.jpg')

    pil_image = Image.fromarray(image)
    face = ImageDraw.Draw(pil_image, 'RGBA')
    face.rectangle((right, top, left, bottom))
    pil_image.save('./testimg/' + imgtitle + '_testimg.jpg')
    pil_image.show()