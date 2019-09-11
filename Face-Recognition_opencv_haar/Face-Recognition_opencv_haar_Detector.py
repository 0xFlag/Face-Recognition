#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import cv2,os
import numpy as np
from PIL import Image 

path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'./trainer/trainer.yml')
cascadePath = r"./Classifiers/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
dataPath = r'./testimg/'

for img in os.listdir(dataPath):
    im = cv2.imread(dataPath + img)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(0,255,0),2)
        if(nbr_predicted==1):
            nbr_predicted='Guido van Rossum'
        elif(nbr_predicted==2):
            nbr_predicted='Chris Wanstrath'
        # 绘制文本
        cv2.putText(im,str(nbr_predicted)+"-"+str(conf), (x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0))
        cv2.imshow('im',im)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()