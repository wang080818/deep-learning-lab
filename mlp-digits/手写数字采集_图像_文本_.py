import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageGrab
import numpy as np
from scipy.spatial.distance import cdist
import os
import glob
from collections import Counter
from datetime import datetime

class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字采集")
        self.root.geometry("400x300")

        self.canvas = tk.Canvas(root, width=100, height=100, bg='white')
        self.canvas.pack(pady=10)

        self.draw = False
        self.start_x, self.start_y = None, None
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_b1_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.label = tk.Label(root, text="标签")
        self.label.pack(pady=5)

        self.textbox2 = tk.Entry(root, width=20)
        self.textbox2.pack(pady=5)

        self.button = tk.Button(root, text="保存图片", command=self.SavePic)
        self.button.pack(pady=10)

        clear_button = tk.Button(root, text="清除画布", command=self.clear_canvas)
        clear_button.pack(pady=10)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.label.config(text="")
        self.textbox2.delete(0,tk.END)

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

    def SavePic(self):
        x1, y1, x2, y2 = self.canvas.bbox(tk.ALL)
        if x1 is None or y1 is None or x2 is None or y2 is None:
            messagebox.showwarning("No Input", "Please draw something first!")
            return

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

        user_text = self.textbox2.get()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{user_text}_{timestamp}.txt"

        subdir = "trainingDigits"
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        width,height=image.size

        with open(subdir+"\\"+filename, 'w') as f:
            for y in range(height):
                for x in range(width):
                    pixel_value = '1' if image.getpixel((x, y)) == 255 else '0'
                    f.write(pixel_value)
                f.write('\n')


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()
