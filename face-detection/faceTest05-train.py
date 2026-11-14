import os
import cv2 as cv
import numpy as np
faces_list = []
labels = []
label = 1
# 遍历trainphotos文件夹下面的图片,检测并提取出人脸信息
for f in os.listdir('./trainphotos'):
    # 读取图片
    img = cv.imread(os.path.join('./trainphotos', f), 0)
    # 提取人脸信息 以下两行代码，如果报错选另一行
    #face_classifier = cv.CascadeClassifier(".venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        continue
    for (x, y, w, h) in faces:
        faces_list.append(img[y:y + h, x:x + w])
        labels.append(label)
    label += 1

# 训练数据并保存
recognizer = cv.face.LBPHFaceRecognizer_create()  # opencv-contrib-python
recognizer.train(faces_list, np.array(labels))  # 一组人脸信息,一组人脸对应标签
recognizer.write('train.yml')
