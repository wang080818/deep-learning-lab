import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import ttk

def load_mnist_data():
    """加载MNIST数据集并返回归一化后的数据"""
    with np.load('mnist.npz') as data:
        x_train = data['x_train'].astype(np.float32) / 255.0
        y_train = data['y_train'].astype(np.int32)
        x_test = data['x_test'].astype(np.float32) / 255.0
        y_test = data['y_test'].astype(np.int32)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """构建三层神经网络模型"""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),      # 输入层
        layers.Dense(128, activation='relu'),      # 第一隐藏层
        layers.Dense(64, activation='relu'),       # 第二隐藏层
        layers.Dense(32, activation='relu'),       # 第三隐藏层
        layers.Dense(10, activation='softmax')     # 输出层
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate():
    """训练和评估模型，更新界面显示"""
    global test_acc

    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # 构建模型
    model = build_model()

    # 训练模型（静默模式）
    model.fit(x_train, y_train,
              epochs=5,
              batch_size=64,
              verbose=0,
              validation_split=0.1)

    # 评估测试集
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # 更新界面显示
    accuracy_label.config(text=f"测试准确率: {test_acc:.4f}")
    progress['value'] = test_acc * 100

# 创建GUI界面
root = tk.Tk()
root.title("MNIST神经网络测试")

# 准确率显示区域
accuracy_frame = tk.Frame(root)
accuracy_frame.pack(pady=20)

accuracy_label = tk.Label(accuracy_frame, text="正在训练模型...")
accuracy_label.pack()

# 进度条
progress = ttk.Progressbar(root, orient="horizontal",
                           length=300, mode="determinate")
progress.pack(pady=10)

# 训练按钮
train_button = tk.Button(root, text="重新训练并评估", command=train_and_evaluate)
train_button.pack(pady=10)

# 初始训练
train_and_evaluate()

# 运行主循环
root.mainloop()
