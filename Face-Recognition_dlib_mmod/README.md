# Face-Recognition_dlib_mmod
</br>
编程语言: Python3</br>
编程模块: OpenCV(4.1.1) dlib(19.8.1)</br>
识别模型: mmod_human_face_detector.dat</br>
</br>
images 待识别图片文件夹</br>
models 识别模型</br>
testimg 识别后存放文件夹</br>
Face-Recognition_dlib_mmod.py</br>
</br>
优点:</br>
1.适用于不同的人脸方向</br>
2.对遮挡鲁棒</br>
3.在GPU上工作得非常快</br>
4.非常简单的训练过程</br>
</br>
缺点:</br>
1.CPU速度很慢</br>
2.不能检测小脸，因为它训练数据的最小人脸尺寸为80×80，但是用户可以用较小尺寸的人脸数据自己训练检测器</br>
3.人脸包围框甚至小于DLib HoG人脸检测器</br>
</br>
吐槽:</br>
建议不要使用这种方法人脸识别，识别甚至比DLib HoG差</br>
</br>
参考:</br>
http://dlib.net/files/</br>
