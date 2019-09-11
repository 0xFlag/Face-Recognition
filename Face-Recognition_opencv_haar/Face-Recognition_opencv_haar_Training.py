#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import cv2,os
import numpy as np
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = path+r"\Classifiers\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
dataPath = path+r'\testimg'

def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
    # images 人脸图像
    images = []
    # labels 分配图像标签
    labels = []
    for image_path in image_paths:
        # 读取图像并转换为灰度
        image_pil = Image.open(image_path).convert('L')
        # 将图像格式转换为numpy数组
        image = np.array(image_pil, 'uint8')
        # 获取图像标签
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
        #nbr=int(''.join(str(ord(c)) for c in nbr))
        print(nbr)
        # 检测图像中的人脸
        faces = faceCascade.detectMultiScale(image)
        # 如果检测到人脸，将人脸添加到图像中，并将标签添加到labels中
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            print(labels)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(500)
    # 返回图像列表和标签列表
    return images, labels

images, labels = get_images_and_labels(dataPath)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
recognizer.save(path+r'\trainer\trainer.yml')
cv2.destroyAllWindows()