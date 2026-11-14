import cv2
def main():
    # 创建一个VideoCapture对象，可用来捕捉视频
    #---------------------下面是摄像头或视频流的二选一方法开始--------------------------------#
    #方法一:捕捉指定路径下的视频
    cap = cv2.VideoCapture("video\\东京街头的人流.mp4")

    #方法二：捕捉默认第一个摄像头，测试时，注释上面行，取消下一行的注释
    #cap = cv2.VideoCapture(0)
    #---------------------摄像头或视频流的二选一方法区域结束--------------------------------#

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("Sorry,捕捉视频没有成功！")
        return
    while True:
        # 逐帧捕获视频
        ret, frame = cap.read()
        # 如果帧读取失败（例如视频结束），则退出循环
        if not ret:
            print("视频播放完了")
            break
        # 显示帧
        cv2.imshow('视频播放', frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty('视频播放', cv2.WND_PROP_VISIBLE) < 1.0:
            break
        # 按esc退格键退出
        if key == 27:
            break
    # 释放VideoCapture对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
