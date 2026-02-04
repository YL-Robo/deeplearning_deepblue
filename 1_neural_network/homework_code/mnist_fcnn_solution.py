"""
MNIST手写数字识别 - 全连接神经网络完整解答
基于 homework_review.pdf 和 mnist_fcnn_exercise.ipynb

作业要求: 补全6处代码
1. sigmoid函数
2. sigmoid导数
3. 权重和偏置初始化
4. 前向传播
5. 调用反向传播
6. 隐藏层误差计算
"""

import numpy as np
from matplotlib import pyplot as plt
import _pickle as cPickle
import gzip
import random
import os
from sklearn.metrics import classification_report


# ============================================
# 数据加载
# ============================================

def load_data():
    """加载MNIST数据集"""
    # 获取脚本所在目录，构建绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../coding_homework/code_homework/MNIST/mnist.pkl.gz')
    f = gzip.open(data_path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """将数字标签转换为10维one-hot向量"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def minist_loader():
    """返回格式化的MNIST数据"""
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 测试集使用one-hot编码，是因为要计算损失函数loss
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # 验证集不需要one-hot编码，只需标签，是因为只关心准确率
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


# ============================================
# 激活函数 (作业补全1 & 2)
# ============================================

def sigmoid(z):
    """
    Sigmoid激活函数: σ(z) = 1/(1+e^(-z))
    将任意实数映射到(0,1)区间
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Sigmoid函数的导数: σ'(z) = σ(z) * (1-σ(z))

    推导:
    σ(z) = 1/(1+e^(-z))
    σ'(z) = e^(-z)/(1+e^(-z))^2 = σ(z) * (1-σ(z))
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


# ============================================
# 全连接神经网络类
# ============================================

class FCN(object):
    """全连接神经网络"""

    def __init__(self, sizes):
        """
        初始化网络
        :param sizes: 列表,每层神经元数量,如[784, 30, 10]
        """
        self._num_layers = len(sizes)

        # 作业补全3: 初始化偏置向量
        # 为隐藏层和输出层生成偏置(输入层不需要偏置)
        # 使用标准正态分布N(0,1)
        self._biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 作业补全3: 初始化权重矩阵
        # W^(l)的维度: (当前层神经元数, 前一层神经元数)
        self._weights = [np.random.randn(y, x)
                         for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        前向传播: 计算网络输出
        公式: a^(l+1) = σ(W^(l) * a^(l) + b^(l))
        """
        # 作业补全4: 实现前向传播
        for w, b in zip(self._weights, self._biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        使用小批量随机梯度下降训练网络
        :param training_data: 训练数据 [(x, y), ...]
        :param epochs: 训练轮数
        :param mini_batch_size: 小批量大小
        :param eta: 学习率
        :param test_data: 测试数据(可选)
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            # 每轮训练前随机打乱数据
            random.shuffle(training_data)
            # 划分小批量
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            # 对每个小批量进行训练
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # 评估性能
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j}: {accuracy}/{n_test} ({100.0*accuracy/n_test:.2f}%)")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        使用一个小批量更新网络参数
        :param mini_batch: 小批量数据 [(x, y), ...]
        :param eta: 学习率
        """
        # 初始化梯度累加器
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]

        # 对每个样本计算梯度并累加
        for x, y in mini_batch:
            # 作业补全5: 调用反向传播算法
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # 累加梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降更新参数
        self._weights = [w - (eta / len(mini_batch)) * nw
                         for w, nw in zip(self._weights, nabla_w)]
        self._biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self._biases, nabla_b)]

    def backprop(self, x, y):
        """
        反向传播算法: 计算损失函数的梯度
        :param x: 输入样本 (784x1)
        :param y: 标签 (10x1 one-hot)
        :return: (nabla_b, nabla_w) 梯度元组
        """
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]

        # === 前向传播 ===
        activation = x
        activations = [x]  # 存储所有激活值
        zs = []  # 存储所有带权输入

        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # === 反向传播 ===
        # 输出层误差: δ^(L) = (a^(L) - y) ⊙ σ'(z^(L))
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 从倒数第二层开始反向传播
        for l in range(2, self._num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            # 作业补全6: 计算隐藏层误差
            # δ^(l) = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^(l))
            delta = np.dot(self._weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        评估网络性能
        :param test_data: 测试数据 [(x, y), ...]
        :return: 正确分类的样本数
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        损失函数对输出层激活值的偏导数
        损失函数: C = 1/2 ||y - a||^2
        偏导数: ∂C/∂a = a - y
        """
        return (output_activations - y)


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    print("正在加载MNIST数据集...")
    training_data, validation_data, test_data = minist_loader()
    print(f"训练集: {len(training_data)}, 验证集: {len(validation_data)}, 测试集: {len(test_data)}")

    # 创建网络: 784输入 -> 30隐藏 -> 10输出
    print("\n创建神经网络 [784, 30, 10]...")
    fc = FCN([784, 30, 10])

    # 训练网络
    print("\n开始训练...")
    fc.train(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

    # 详细评估
    print("\n生成分类报告...")
    predictions = [np.argmax(fc.feedforward(x)) for x, _ in test_data]
    labels = [y for _, y in test_data]
    print(classification_report(labels, predictions))

    print("\n训练完成!")

