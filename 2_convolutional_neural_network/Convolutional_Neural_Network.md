# 第一部分：全连接网络回顾

**(PDF Page 1 - 20)**

## 1. 理论核心

### 1.1 全连接网络结构

* **输入层**：以MNIST为例，输入为 $28 \times 28 = 784$ 维向量。
* **隐藏层**：每个神经元与前一层所有神经元全连接，计算公式：

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})$$

* **输出层**：10个神经元对应数字0-9，通过Softmax输出概率分布。

### 1.2 反向传播回顾

* **输出层误差**：$\delta^{(L)} = (a^{(L)} - y) \odot \sigma'(z^{(L)})$
* **隐藏层误差**：$\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot \sigma'(z^{(l)})$
* **参数梯度**：$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l+1)} \cdot a^{(l)T}$，$\frac{\partial C}{\partial b^{(l)}} = \delta^{(l+1)}$

### 1.3 全连接网络的局限性

* **参数爆炸**：当图片尺寸增大（如 $256 \times 256$），输入层变为65536个神经元，第一层权重矩阵参数量高达数百万，极易过拟合。
* **解决方案**：使用**卷积神经网络 (CNN)**，利用**局部连接**和**权值共享**大幅减少参数量。

## 2. 辅助工具与代码

Sigmoid函数及其导数实现：

```python
def sigmoid(z):
    """Sigmoid激活函数: σ(z) = 1/(1+e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Sigmoid导数: σ'(z) = σ(z) * (1-σ(z))"""
    return sigmoid(z) * (1.0 - sigmoid(z))
```

## 3. 课后作业解答（course02_code.py 补全代码）

**3.1 Sigmoid函数及其导数**

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))
```

**3.2 网络初始化 (__init__)**

```python
# 偏置向量: 对每个隐藏层和输出层生成随机偏置
self._biases = [np.random.randn(y, 1) for y in sizes[1:]]

# 权重矩阵: 连接相邻两层，维度为 (当前层神经元数, 前一层神经元数)
self._weights = [np.random.randn(y, x)
                 for x, y in zip(sizes[:-1], sizes[1:])]
```

**3.3 前向传播 (feedforward)**

```python
for w, b in zip(self._weights, self._biases):
    a = sigmoid(np.dot(w, a) + b)
```

**3.4 调用反向传播 (update_mini_batch)**

```python
delta_nabla_b, delta_nabla_w = self.backprop(x, y)
```

**3.5 隐藏层误差计算 (backprop)**

```python
# δ^(l) = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^(l))
delta = np.dot(self._weights[-l+1].transpose(), delta) * sp
```

---

# 第二部分：卷积神经网络基础

**(PDF Page 21 - 45)**

## 1. 理论核心

### 1.1 为什么需要CNN？

* **局部连接**：每个神经元只与输入的一个局部区域连接，而非全部。例如一个 $5 \times 5$ 的卷积核只关注 $5 \times 5$ 的局部区域。
* **权值共享**：同一个卷积核在整张图片上滑动，所有位置共享同一组参数，大幅减少参数量。
* **平移不变性**：无论特征出现在图片的哪个位置，同一个卷积核都能检测到。

### 1.2 卷积操作

* **卷积核 (Kernel/Filter)**：一个小的权重矩阵，在输入上滑动，对每个局部区域做元素乘积求和。
* **特征图 (Feature Map)**：卷积核滑过整个输入后得到的输出。
* **输出尺寸公式**：

$$\text{feature\_w} = \left\lfloor \frac{W - K}{S} \right\rfloor + 1$$

其中 $W$ 为输入宽度，$K$ 为卷积核大小，$S$ 为步长。

### 1.3 img2col 加速技巧

将卷积操作转化为矩阵乘法以加速计算：
* 将输入中每个待卷积的局部区域展平为一行，堆叠成矩阵，形状为 $(feature\_w^2, K^2 \times C)$
* 将卷积核展平为矩阵，形状为 $(K^2 \times C, N)$，其中 $N$ 为卷积核个数
* 卷积 = 两个矩阵相乘

### 1.4 池化 (Pooling)

* **最大池化 (Max Pooling)**：在 $2 \times 2$ 窗口中取最大值，步长为2，将特征图尺寸减半。
* **作用**：降低特征图分辨率，减少参数量，增强平移不变性。
* **反向传播**：梯度只传递到最大值所在位置，需要在前向传播时记录最大值位置的mask。

### 1.5 激活函数：ReLU

$$\text{ReLU}(x) = \max(0, x)$$

* 相比Sigmoid，ReLU计算简单，不存在梯度消失问题。
* 反向传播：输入大于0时梯度为1，否则为0。

## 2. 辅助工具与代码

img2col 函数将卷积转化为矩阵乘法：

```python
def img2col(x, ksize, step):
    wx, hx, cx = x.shape
    feature_w = (wx - ksize) // step + 1
    image_col = np.zeros((feature_w * feature_w, ksize * ksize * cx))
    num = 0
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] = x[i*step:i*step+ksize, j*step:j*step+ksize, :].reshape(-1)
            num += 1
    return image_col
```

## 3. 课后作业解答（course03_code.py 补全代码）

**3.1 img2col 函数补全**

```python
num = 0
for i in range(feature_w):
    for j in range(feature_w):
        image_col[num] = x[i*step:i*step+ksize, j*step:j*step+ksize, :].reshape(-1)
        num += 1
```

*解释*：遍历所有空间位置，将每个 $ksize \times ksize \times channel$ 的局部区域展平为一行。

**3.2 Pool.forward 最大值提取**

```python
feature[bi, i, j, ci] = np.max(x[bi, i*2:i*2+2, j*2:j*2+2, ci])
```

*解释*：在 $2 \times 2$ 窗口中取最大值，同时通过 `argmax` 记录最大值位置用于反向传播。

---

# 第三部分：CNN前向与反向传播

**(PDF Page 46 - 70)**

## 1. 理论核心

### 1.1 网络架构（LeNet风格）

本课程实现的CNN架构如下：

$$\text{Input}(28 \times 28 \times 1) \rightarrow \text{Conv1}(5 \times 5, 6) \rightarrow \text{ReLU} \rightarrow \text{Pool}(2 \times 2)$$
$$\rightarrow \text{Conv2}(5 \times 5, 16) \rightarrow \text{ReLU} \rightarrow \text{Pool}(2 \times 2) \rightarrow \text{FC}(256 \rightarrow 10) \rightarrow \text{Softmax}$$

**各层维度变化**：

| 层 | 输出尺寸 | 说明 |
|---|---------|------|
| 输入 | $28 \times 28 \times 1$ | MNIST灰度图 |
| Conv1 | $24 \times 24 \times 6$ | 卷积核 $5 \times 5$，6个 |
| Pool1 | $12 \times 12 \times 6$ | $2 \times 2$ 最大池化 |
| Conv2 | $8 \times 8 \times 16$ | 卷积核 $5 \times 5$，16个 |
| Pool2 | $4 \times 4 \times 16$ | $2 \times 2$ 最大池化 |
| Flatten | $256$ | $4 \times 4 \times 16 = 256$ |
| FC | $10$ | 全连接输出层 |

### 1.2 卷积层前向传播

利用 img2col 加速：

1. 对每个样本调用 `img2col` 将局部区域展平为矩阵
2. 将卷积核 reshape 为二维矩阵 $(K^2 \times C_{in}, C_{out})$
3. 矩阵乘法得到输出特征图

### 1.3 卷积层反向传播

**参数梯度**：
* $\frac{\partial L}{\partial K}$：利用 `image_col` 和 `delta_col` 的矩阵乘法计算
* $\frac{\partial L}{\partial b}$：对 delta 在空间维度上求和

**误差反向传递**：
* 将卷积核旋转180度（`np.flipud` + `np.fliplr`）
* 对 delta 进行 padding 使尺寸匹配
* 用旋转后的卷积核与 padded delta 做卷积，得到传递给前一层的误差

### 1.4 池化层反向传播

* 前向传播时记录每个 $2 \times 2$ 窗口中最大值的位置（mask）
* 反向传播时，梯度只传递到最大值位置，其余位置梯度为0

## 2. 辅助工具与代码

Softmax 与交叉熵损失：

```python
class Softmax(object):
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta
```

## 3. 课后作业解答（course03_code.py 补全代码续）

**3.3 Conv.forward 前向传播补全**

```python
for i in range(bx):
    image_col = img2col(self.x[i], wk, self.step)
    feature[i] = (np.dot(image_col, kernel) + self.b).reshape(feature_w, feature_w, nk)
    self.image_col.append(image_col)
```

*解释*：对每个样本，用 `img2col` 展开输入，与 reshape 后的卷积核做矩阵乘法，加上偏置后 reshape 为特征图。

**3.4 Conv.backward 梯度计算与参数更新补全**

```python
delta_col = delta.reshape(bd, -1, cd)
for i in range(bx):
    self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
self.k_gradient /= bx
self.b_gradient += np.sum(delta_col, axis=(0, 1))
self.b_gradient /= bx
```

*解释*：利用前向传播保存的 `image_col` 与 delta 做矩阵乘法得到卷积核梯度，对 batch 取平均。

**3.5 Conv.backward 误差反向传递（卷积核旋转180度）补全**

```python
k_180 = np.flipud(self.k)
k_180 = np.fliplr(k_180)
k_180_col = k_180.swapaxes(2, 3).reshape(-1, ck)
```

*解释*：将卷积核上下翻转再左右翻转实现180度旋转，然后交换输入输出通道维度并 reshape 为二维矩阵，用于与 padded delta 做矩阵乘法。

**3.6 Linear.backward 全连接层反向传播补全**

```python
# 梯度计算
batch_size = self.x.shape[0]
self.W_gradient = np.dot(self.x.T, delta) / batch_size
self.b_gradient = np.sum(delta, axis=0) / batch_size

# 反向传播
delta_backward = np.dot(delta, self.W.T)

# 参数更新
self.W -= self.W_gradient * learning_rate
self.b -= self.b_gradient * learning_rate
```

---

# 第四部分：CNN编程作业解答

## 1. 作业要求

根据 `cnn_exercise.ipynb` 中的要求，需要完成以下任务：

**代码补全任务（4处）**：
1. `img2col` 函数补全 —— 理解卷积转矩阵乘法的过程
2. `Conv` 类中的前向过程
3. `Conv` 类中后向过程的权重更新与误差反向传播
4. `Pool` 函数中的最大位置 mask 的计算过程

**拓展实验任务（2处）**：
1. 修改卷积和池化的组合方式（如卷积后接卷积再接池化），观察对结果的影响
2. 增加 conv3 和 pool3 实例，或修改卷积核大小和个数

## 2. 完整代码

完整实现见 `coding_examples/course03_code.py`，编程练习见 `homework_code/cnn_exercise.ipynb`。

## 3. 关键点总结

* **img2col**：将卷积操作转化为矩阵乘法，是CNN高效实现的核心技巧。每个局部区域展平为一行，所有区域堆叠成矩阵。
* **卷积前向传播**：`img2col(input) × kernel_reshaped + bias`，结果 reshape 为特征图。
* **卷积反向传播**：梯度计算用 `image_col.T × delta_col`；误差传递需要将卷积核旋转180度后与 padded delta 做卷积。
* **池化反向传播**：梯度只传递到前向传播时最大值所在的位置。
* **维度追踪**：在整个网络中跟踪张量维度变化是调试的关键。
* **权重初始化**：使用 $\sqrt{3 \times C_{in} \times K_w \times K_h / C_{out}}$ 作为缩放因子，避免梯度爆炸或消失。
