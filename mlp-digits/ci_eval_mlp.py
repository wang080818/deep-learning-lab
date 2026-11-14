import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def main():
    # 云端无本机 trainingDigits 样本，这里用 sklearn 内置 8x8 digits 数据集做等价评测
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52)

    mlp = MLPClassifier(hidden_layer_sizes=(200, 300, 300),
                         max_iter=1500, random_state=52)
    mlp.fit(X_train, y_train)
    acc = mlp.score(X_test, y_test)
    print(f"[CI] MLP digits 测试集准确率: {acc:.4f}")


if __name__ == '__main__':
    main()
