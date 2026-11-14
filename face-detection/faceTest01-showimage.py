#导入Opencv库
import cv2
# 读取图片，图片位置放在和本文件同目录下的images文件下的image.jpg图片
image = cv2.imread('./images/image.jpg')
# 注意图片路径，你也可以根据需要修改为你想要的图片
# 显示图片
cv2.imshow("images", image)
# 等待按键事件
cv2.waitKey(0)
# 关闭所有窗口
