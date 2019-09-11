#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

# 1.导入库
import cv2
import os

# 2.加载图片，加载模型
# 待检测的图片路径
imagepath = r"./testimg/test.jpg"
# 获取训练好的人脸的参数数据，这里直接使用默认值
face_cascade = cv2.CascadeClassifier(r"./Classifiers/haarcascade_frontalface_default.xml")
# haarcascade_frontalface_default.xml
# haarcascade_frontalface_alt.xml
# haarcascade_frontalface_alt2.xml
# 读取图片
image = cv2.imread(imagepath)

# 3.对图片灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4.检测人脸，探测图片中的人脸
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(5,5),flags=cv2.CASCADE_SCALE_IMAGE)
# flags=cv2.cv.CV_HAAR_SCALE_IMAGE
# flags=cv2.CASCADE_SCALE_IMAGE OpenCV3.1+

if len(faces) == 0:
	print("未发现人脸")
else:
	print("发现{0}个人脸!".format(len(faces)))
	# 5.标记人脸
	for(x,y,w,h) in faces:
	# 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

		# 6.显示图片
		cv2.imshow("Find Faces!", image)

		# 7.保存图片
		imgtitle = os.path.splitext(imagepath)[0]
		cv2.imwrite("./dataimg/" + imgtitle + "_dataimg.jpg",image)
		cv2.imwrite("./faceimg/" + imgtitle + "_faceimg.jpg",image[y:y+h,x:x+w])

		# 8.暂停窗口
		cv2.waitKey()
		#cv2.waitKey(5000)

		# 9.销毁窗口
		#cv2.destroyAllWindows()