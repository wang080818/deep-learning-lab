import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageGrab
import numpy as np
import os
import glob
from sklearn.neural_network import MLPClassifier

class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别_神经网络")
        self.root.geometry("300x300")

        self.canvas = tk.Canvas(root, width=100, height=100, bg='white')
        self.canvas.pack(pady=10)

        self.draw = False
        self.start_x, self.start_y = None, None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_b1_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.button = tk.Button(root, text="开始识别", command=self.recognize)
        self.button.pack(pady=10)

        self.label = tk.Label(root, text="", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # 创建按钮
        clear_button = tk.Button(root, text="清空画布", command=self.clear_canvas)
        clear_button.pack(pady=20)

        self.training_data = self.load_training_data()
        self.mlp = self.train_mlp()

    def clear_canvas(self):
        # 清空画布内容
        self.canvas.delete("all")
        self.label.config(text="")

    def on_button_press(self, event):
        self.draw = True
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasx(event.y)

    def on_b1_motion(self, event):
        if self.draw:
            curX, curY = self.canvas.canvasx(event.x), self.canvas.canvasx(event.y)
            self.canvas.create_line(self.start_x, self.start_y, curX, curY, smooth=True, width=12, capstyle=tk.ROUND, fill='black')
            self.start_x, self.start_y = curX, curY

    def on_button_release(self, event):
        self.draw = False

    def save_canvas_as_image(self,root, canvas, filename):
        bbox = (root.winfo_x()+canvas.winfo_x(), root.winfo_y()+canvas.winfo_y(), root.winfo_x()+canvas.winfo_x() + canvas.winfo_width(), root.winfo_y()+canvas.winfo_height())
        img = ImageGrab.grab(bbox)
        img.save(filename)

    def recognize(self):
        x1, y1, x2, y2 = self.canvas.bbox(tk.ALL)
        if x1 is None or y1 is None or x2 is None or y2 is None:
            messagebox.showwarning("No Input", "Please draw something first!")
            return

        self.save_canvas_as_image(self.root, self.canvas, 'canvas_image.png')

        image = Image.new('RGB', (x2 - x1, y2 - y1), 'white')
        draw = ImageDraw.Draw(image)

        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'line':
                coords = self.canvas.coords(item)
                adjusted_coords = [coords[0]-x1,coords[1]-y1,coords[2]-x1,coords[3]-y1]
                draw.line(adjusted_coords, fill='black', width=9)

        image = image.convert('L').point(lambda p: 255 if p > 128 else 0)
        image = ImageOps.invert(image)
        image = image.resize((32, 32), Image.NEAREST)

        width,height=image.size

        with open('test1.txt', 'w') as f:
            for y in range(height):
                for x in range(width):
                    pixel_value = '1' if image.getpixel((x, y)) == 255 else '0'
                    f.write(pixel_value)
                f.write('\n')

        test_image_data = self.load_image_data('test1.txt')
        test_image_flat = test_image_data.flatten().reshape(1, -1)
        predicted_label = self.mlp.predict(test_image_flat)[0]

        self.label.config(text=f"预测数字: {predicted_label}")

    def load_training_data(self):
        training_folder = 'trainingDigits'
        training_files = glob.glob(os.path.join(training_folder, '*.txt'))
        return training_files

    def load_image_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data = np.array([[int(pixel) for pixel in line.strip()] for line in lines], dtype=int)
        return data

    def train_mlp(self):
        X = []
        y = []
        for file in self.training_data:
            label, _ = os.path.splitext(os.path.basename(file))[0].split('_')
            label = int(label)
            training_image_data = self.load_image_data(file)
            X.append(training_image_data.flatten())
            y.append(label)
        X = np.array(X)
        y = np.array(y)

        mlp = MLPClassifier(hidden_layer_sizes=(200,300,300), max_iter=1500, random_state=52)
        mlp.fit(X, y)
        return mlp

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()
