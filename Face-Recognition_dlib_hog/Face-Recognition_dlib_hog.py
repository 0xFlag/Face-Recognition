#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import cv2
import dlib
import time
import sys
import os

def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):
    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    print(frameWidth, frameHeight, inWidth, inHeight)

    bboxes = []
    for faceRect in faceRects:

        cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),
                  int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/600)), 4)
    return frameDlibHog, bboxes

if __name__ == "__main__" :
    hogFaceDetector = dlib.get_frontal_face_detector()

    frame_count = 0
    tt_dlibHog = 0
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

        # 计算DLIB HoG FPS
        t = time.time()
        outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
        tt_dlibHog += time.time() - t
        fpsDlibHog = frame_count / tt_dlibHog

        # DLIB HoG FPS显示
        text = "DLIB HoG FPS : {:.2f}".format(fpsDlibHog)
        cv2.putText(outDlibHog, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", outDlibHog)

        if frame_count == 1:
            tt_dlibHog = 0

        # 识别图片自动保存
        imgsplit = os.path.split(filepath)[1]
        imgtitle = os.path.splitext(imgsplit)[0]
        cv2.imwrite("./testimg/" + imgtitle + "_testimg.jpg",outDlibHog)

        # 显示停留时间
        cv2.waitKey(500)
