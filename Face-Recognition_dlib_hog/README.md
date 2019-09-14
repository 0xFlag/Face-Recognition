# Face-Recognition_dlib_hog
</br>
编程语言: Python3</br>
编程模块: OpenCV(4.1.1) dlib(19.8.1)</br>
</br>
images 待识别图片文件夹</br>
testimg 识别后存放图片文件夹</br>
Face-Recognition_dlib_hog.py 识别图片</br>
Face-Recognition_dlib_hog_cap.py 摄像头识别</br>
</br>
优点:</br>
1.CPU上最快的方法</br>
2.适用于正面和略微非正面的人脸</br>
3.与其他三个相比模型很小</br>
4.在小的遮挡下仍可工作</br>
</br>
缺点:</br>
1.不能检测小脸，因为它训练数据的最小人脸尺寸为80×80，但是用户可以用较小尺寸的人脸数据自己训练检测器</br>
2.边界框通常排除前额的一部分甚至下巴的一部分</br>
3.在严重遮挡下不能很好地工作</br>
4.不适用于侧面和极端非正面，如俯视或仰视</br>
</br>
吐槽:</br>
建议不要用这种来人脸识别</br>
