import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tkinter as tk

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")

        # 初始化绘图数组（28x28，1代表空白，0代表笔画）
        self.drawing_array = np.ones((28, 28), dtype=np.float32)

        # 创建界面组件
        self.create_widgets()

        # 加载并训练模型
        self.model = self.train_model()

    def create_widgets(self):
        """创建画布和按钮"""
        # 绘制区域（280x280像素对应28x28网格）
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        # 绑定绘图事件
        self.canvas.bind('<B1-Motion>', self.draw)

        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        # 识别按钮
        self.recognize_btn = tk.Button(
            button_frame, text="识别", command=self.recognize
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        # 清除按钮
        self.clear_btn = tk.Button(
            button_frame, text="清除", command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # 结果显示标签
        self.result_label = tk.Label(button_frame, text="识别结果：")
        self.result_label.pack(side=tk.LEFT, padx=5)

    def draw(self, event):
        """处理鼠标绘图事件"""
        x = event.x
        y = event.y

        # 将像素坐标转换为网格坐标（每个网格10x10像素）
        col = x // 10
        row = y // 10

        if 0 <= col < 28 and 0 <= row < 28:
            # 更新绘图数组（0表示已绘制）
            self.drawing_array[row, col] = 0

            # 在画布上绘制黑色方块
            x1 = col * 10
            y1 = row * 10
            x2 = x1 + 10
            y2 = y1 + 10
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='')

    def recognize(self):
        """执行识别操作"""
        # 预处理图像数据（取反使笔画变为白色）
        img = 1 - self.drawing_array.reshape(1, 28, 28)

        # 进行预测
        prediction = self.model.predict(img)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        # 显示结果
        self.result_label.config(
            text=f"识别结果：{predicted_label}（置信度：{confidence:.2f}）"
        )

    def clear_canvas(self):
        """清除画布和绘图数组"""
        self.canvas.delete("all")
        self.drawing_array.fill(1)
        self.result_label.config(text="")

    def train_model(self):
        """加载数据并训练神经网络模型"""
        # 加载MNIST数据集
        with np.load('mnist.npz') as data:
            x_train = data['x_train'].astype(np.float32) / 255.0
            y_train = data['y_train']

        # 构建三层神经网络
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # 编译模型
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型（静默模式）
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        return model

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
