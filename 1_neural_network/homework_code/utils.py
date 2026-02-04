"""
工具函数模块
提供数据加载、可视化等辅助功能
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def load_mnist():
    """
    加载MNIST数据集

    Returns:
        X_train, y_train, X_test, y_test: 训练集和测试集
    """
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

    # 划分训练集和测试集
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # 归一化到[0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def visualize_samples(X, y, n_samples=10):
    """
    可视化样本

    Args:
        X: 图像数据 (n_samples, 784)
        y: 标签
        n_samples: 显示样本数量
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Label: {y[i]}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("样本可视化已保存到 mnist_samples.png")


def plot_training_history(history):
    """
    绘制训练历史

    Args:
        history: 包含loss和accuracy的字典
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['accuracy'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("训练历史已保存到 training_history.png")


def visualize_predictions(X, y_true, y_pred, n_samples=10):
    """
    可视化预测结果

    Args:
        X: 图像数据
        y_true: 真实标签
        y_pred: 预测标签
        n_samples: 显示样本数量
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            color = 'green' if y_true[i] == y_pred[i] else 'red'
            ax.set_title(f'True: {y_true[i]}, Pred: {y_pred[i]}', color=color)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 predictions.png")


def one_hot_encode(y, num_classes=10):
    """
    One-hot编码

    Args:
        y: 标签数组 (n_samples,)
        num_classes: 类别数量

    Returns:
        one_hot: (n_samples, num_classes)
    """
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot
