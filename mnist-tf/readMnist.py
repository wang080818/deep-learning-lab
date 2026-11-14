import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_mnist_data():
    """加载MNIST数据集并返回归一化后的训练数据"""
    with np.load('mnist.npz') as data:
        x_train = data['x_train'].astype(np.float32) / 255.0  # 归一化到[0,1]
        y_train = data['y_train']
    return x_train, y_train

def show_new_images():
    """生成并显示新的随机样本"""
    indices = np.random.choice(len(x_train), size=20, replace=False)
    sample_images = x_train[indices]
    sample_labels = y_train[indices]

    fig.clf()  # 清除之前的图像

    # 创建4行5列的子图布局
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1)
        ax.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Label: {sample_labels[i]}", fontsize=8)
        ax.axis('off')

    fig.tight_layout()
    canvas.draw()  # 更新画布

# 加载数据
x_train, y_train = load_mnist_data()

# 创建GUI窗口
root = tk.Tk()
root.title("MNIST 数据集")

# 创建按钮
button = ttk.Button(root, text="显示新样本", command=show_new_images)
button.pack(pady=10)

# 创建Matplotlib画布
fig = Figure(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 初始化显示
show_new_images()

# 运行主循环
root.mainloop()
