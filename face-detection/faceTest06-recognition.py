import cv2 as cv
# 读取图片，并转成灰度图像
img = cv.imread('images/my.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 提取图片中的人脸特征信息 以下两行代码，如果报错选另一行
#face_classifier = cv.CascadeClassifier(".venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray)
# 加载识别器和训练数据
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('train.yml')
# 遍历图片中的人脸
for x, y, w, h in faces:
    # 识别图片中的人脸，返回标签和置信度
    img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
    if confidence > 85:
        name = 'success'
    else:
        name = 'unkown' if img_id == 1 else 'unknow'
    # 标出识别出的人名，用圆圈出人脸
    cv.putText(
        img=img, org=(x, y), text=name,
        fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
        color=(0, 255, 0), thickness=1
    )
    cv.circle(
        img=img, center=(x + w // 2, y + h // 2), radius=w // 2,
        color=(255, 0, 0), thickness=1
    )
# 展示标记后的图片
cv.imshow('face', img)
while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27 or cv.getWindowProperty('face', cv.WND_PROP_VISIBLE) < 1:
        break
# 释放内存
cv.destroyAllWindows()
