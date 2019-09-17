#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import face_recognition
import cv2

# Open the input movie file
input_movie = cv2.VideoCapture("hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize variables
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

    print("Writing frame {} / {}".format(frame_number, length))

    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite("./videoimg/" + str(frame_number) + ".jpg",frame)

        face_image = frame[top:bottom, left:right]
        cv2.imwrite("./faceimg/" + str(frame_number) + ".jpg",face_image)

# All done!
input_movie.release()
cv2.destroyAllWindows()
