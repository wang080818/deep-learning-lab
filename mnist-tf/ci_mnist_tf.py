import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=2,
              validation_split=0.1)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[CI] MNIST 测试集准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()
