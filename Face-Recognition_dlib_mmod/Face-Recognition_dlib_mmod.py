#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import cv2
import dlib
import time
import sys
import os

def detectFaceDlibMMOD(detector, frame, inHeight=300, inWidth=0):

    frameDlibMMOD = frame.copy()
    frameHeight = frameDlibMMOD.shape[0]
    frameWidth = frameDlibMMOD.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

    frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibMMODSmall, 0)

    print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.rect.left()*scaleWidth), int(faceRect.rect.top()*scaleHeight),
                  int(faceRect.rect.right()*scaleWidth), int(faceRect.rect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/600)), 4)
    return frameDlibMMOD, bboxes

if __name__ == "__main__" :

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    # 识别模型加载
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")

    frame_count = 0
    tt_dlibMmod = 0
    frame_count += 1

    #dir="images"
    #files=os.listdir(dir)
    #filepath=dir+"/img.jpg"
    #frame=cv2.imread(filepath)

    dir="images"
    files=os.listdir(dir)
    for file in files:
        filepath=dir+"/"+file
        frame=cv2.imread(filepath)

        t = time.time()
        outDlibMMOD, bboxes = detectFaceDlibMMOD(dnnFaceDetector,frame)
        tt_dlibMmod += time.time() - t
        fpsDlibMmod = frame_count / tt_dlibMmod

        label = "DLIB MMOD FPS : {:.2f}".format(fpsDlibMmod)
        cv2.putText(outDlibMMOD, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", outDlibMMOD)

        # 识别图片自动保存
        imgsplit = os.path.split(filepath)[1]
        imgtitle = os.path.splitext(imgsplit)[0]
        cv2.imwrite("./testimg/" + imgtitle + "_testimg.jpg",outDlibMMOD)

        if frame_count == 1:
        	tt_dlibMmod = 0

        # 显示停留时间
        cv2.waitKey(10)
