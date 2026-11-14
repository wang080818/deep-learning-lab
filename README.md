# 实验8 深度学习

课程实验代码整理，托管到 GitHub 并接入 GitHub Actions 自动训练评测。

## 目录结构

- `mnist-tf/` — TensorFlow 三层全连接网络识别 MNIST 手写数字（含 Tkinter 交互界面）
- `mlp-digits/` — sklearn MLP 手写数字识别（手写采集 + 识别 GUI）
- `face-detection/` — OpenCV 摄像头人脸检测示例（需本机摄像头，不进 CI）

## 本地运行

```bash
pip install -r requirements.txt
# MNIST 训练+准确率界面
python mnist-tf/recognizeRateMnist.py
# MLP 手写识别（同目录需放 trainingDigits 样本文件夹）
python mlp-digits/手写数字识别_神经网络.py
```

## GitHub Actions 自动评测

推送到 main，或在仓库 Actions 页手动触发（Run workflow）：

- `mnist` 任务：云端自动下载 MNIST，三层全连接网络训练 5 轮，打印测试集准确率
- `mlp` 任务：用 sklearn 内置 digits 数据集训练 MLP，打印测试准确率
- `face-detection` 涉及摄像头与个人照片，不在云端运行

> 说明：原始实验的 1943 个 trainingDigits 样本、mnist.npz、人脸照片与视频未上传仓库（隐私与体积原因），本机运行时自行放入对应目录即可。