#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import cv2
import sys,os
import time

def detectFaceOpenCVDnn(net, image):
    frameOpencvDnn = image.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)

    # 计算人脸识别时间
    start=time.time()
    detections = net.forward()
    end=time.time()
    print(end-start)

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])

            # 显示识别百分比
            text = "{:.2f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            # 1.原始图片 2.人脸坐标原点 3.标记的高度 4.线的颜色 5.线宽
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/600)), 8)
            cv2.putText(frameOpencvDnn, text, (x1, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            global face
            face = frameOpencvDnn[y1:y2,x1:x2]
    return frameOpencvDnn, bboxes

if __name__ == "__main__" :
    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )

    # 加载人脸识别模型
    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "./models/opencv_face_detector_uint8.pb"
        configFile = "./models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    # 批量识别人脸
    #dir="images"
    #files=os.listdir(dir)
    #for file in files:
        #filepath=dir+"/"+file
        #image=cv2.imread(filepath)

    # 识别图片加载
    dir="images"
    files=os.listdir(dir)
    filepath=dir+"/img.jpg"
    image=cv2.imread(filepath)

    frame_count = 0
    tt_opencvDnn = 0

    # OpenCV DNN FPS显示
    frame_count += 1
    t = time.time()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,image)
    tt_opencvDnn += time.time() - t
    fpsOpencvDnn = frame_count / tt_opencvDnn
    text = "OpenCV DNN FPS : {:.2f}".format(fpsOpencvDnn)
    cv2.putText(outOpencvDnn, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Face Detection Comparison", outOpencvDnn)

    # 识别图片自动保存
    imgsplit = os.path.split(filepath)[1]
    imgtitle = os.path.splitext(imgsplit)[0]
    cv2.imwrite("./testimg/" + imgtitle + "_testimg.jpg",outOpencvDnn)
    cv2.imwrite("./faceimg/" + imgtitle + "_faceimg.jpg",face)

    # 显示停留时间
    cv2.waitKey()