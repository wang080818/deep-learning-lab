import cv2 as cv
import numpy as np
def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #face_detector = cv.CascadeClassifier('.venv/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # 二选一，如果此行代码报错无法使用，则使用上行代码
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray, 1.2, 6)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)
print("--------- Python OpenCV Tutorial ---------")
#---------------------下面是可以修改为摄像头或视频流人脸中的其中一项---------start
#检测摄像头人脸
#capture = cv.VideoCapture(0)
#检测视频流人脸
capture = cv.VideoCapture("video\\东京街头的人流.mp4")
#---------------------下面是可以修改为摄像头或视频流人脸中的其中一项---------end
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
while(True):
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)#左右翻转
    face_detect_demo(frame)
    key = cv.waitKey(1) & 0xFF
    if key == 27 or cv.getWindowProperty('result', cv.WND_PROP_VISIBLE) < 1:
        break
capture.release()
cv.destroyAllWindows()
