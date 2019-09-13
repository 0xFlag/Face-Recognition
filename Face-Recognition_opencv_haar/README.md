# Face-Recognition_opencv_haar
</br>
编程语言: Python3</br>
编程模块: OpenCV(4.1.1) numpy(1.17.1) PIL(pillow6.1.0)</br>
</br>
Face-Recognition_opencv_haar.py          单张图片人脸识别</br>
Face-Recognition_opencv_haar_Batch.py    批量图片人脸识别</br>
Face-Recognition_opencv_haar_Detector.py 通过训练文件人脸识别标签</br>
Face-Recognition_opencv_haar_Training.py 人脸识别训练</br>
</br>
Classifiers 存放OpenCV官方识别人脸XML文件</br>
dataimg     存放识别后完整图片</br>
faceimg     存放脸部图片</br>
testimg     存放需要检测图片的文件</br>
trainer     存放训练后生成的yml文件</br>
</br>
识别模型:</br>
haarcascade_frontalface_default.xml</br>
haarcascade_frontalface_alt.xml</br>
haarcascade_frontalface_alt2.xml</br>
</br>
注意:</br>
1.OpenCV版本不一样部分代码不要搞错</br>
</br>
以下则是OpenCV缺点:</br>
1.会出现大量的把非人脸预测为人脸的情况</br>
2.不适用于非正面人脸图像</br>
3.不抗遮挡</br>
</br>
优点:</br>
1.几乎可以在CPU上实时工作</br>
2.简单的架构</br>
3.可以检测不同比例的人脸</br>
</br>
参考:</br>
https://github.com/opencv/opencv/tree/master/data/haarcascades</br>
https://github.com/thecodacus/Face-Recognition</br>
https://github.com/nazmiasri95/Face-Recognition</br>
