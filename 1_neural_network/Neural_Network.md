# 第一部分：极大似然估计与逻辑回归

**(PDF Page 3 - 16)**

## 1. 理论核心

### 1.1 什么是极大似然估计 (MLE)?

* **定义**：利用已知的样本结果（观测值），反推最有可能导致这些结果出现的模型参数。
* **核心区别**：
  * **概率函数 (Probability)**：参数已知 $\rightarrow$ 预测数据发生的可能性。
  * **似然函数 (Likelihood)**：数据已知 $\rightarrow$ 推测参数的可能性。
* *通俗理解*：如果是”已知黑球白球比例算抓到白球的概率”，这是概率问题；如果是”抓了一把球看颜色，反推桶里黑白比例”，这是极大似然问题。

### 1.2 逻辑回归 (Logistic Regression)

* **模型**：虽然叫回归，但本质是**分类算法**。
* **核心公式**：使用 **Sigmoid 函数** 将线性输出 $(-\infty, +\infty)$ 压缩到 $(0, 1)$ 之间，表示概率：

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b$$

* **损失函数**：对数似然损失（交叉熵）。为了求导方便，我们对似然函数取对数并取反（求极小值）：

$$L(w, b) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

## 2. 辅助工具与代码

为了理解 Sigmoid 函数如何将数值”挤压”成概率，我们可以运行以下代码：

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function: Mapping (-inf, inf) to (0, 1)")
plt.grid(True)
plt.show()

```

## 3. 课后作业 I & II 解答

**(PDF Page 8, 15)**

**作业 I (Page 8)**

* **题1**：数据已知（抽样100次70白），求参数（白球比例）。这是由果推因。
  * **答案**：B. 似然函数
* **题2**：参数已知（白球占比70%），求数据发生的可能性。这是由因推果。
  * **答案**：A. 概率函数

**作业 II (Page 15)**

* **问题**：证明为什么负梯度方向是损失函数下降的方向？
* **解答**：

**1. 基于一阶泰勒展开的线性近似**

假设损失函数 $f(\mathbf{x})$ 在点 $\mathbf{x}$ 处连续可微。当自变量有一个微小增量 $\Delta \mathbf{x}$ 时，根据一阶泰勒展开，函数值的变化量 $\Delta f$ 可以近似表示为：

$$\Delta f = f(\mathbf{x} + \Delta \mathbf{x}) - f(\mathbf{x}) \approx \nabla f(\mathbf{x})^T \Delta \mathbf{x}$$

其中 $\nabla f(\mathbf{x})$ 是函数在点 $\mathbf{x}$ 处的梯度向量。为了使函数值下降，我们需要确保 $\Delta f < 0$，即：

$$\nabla f(\mathbf{x})^T \Delta \mathbf{x} < 0$$

**2. 向量点积与几何分析**

根据向量点积的定义，上述公式可以写成：

$$\Delta f \approx \|\nabla f(\mathbf{x})\| \cdot \|\Delta \mathbf{x}\| \cdot \cos(\theta)$$

其中 $\theta$ 是梯度向量 $\nabla f(\mathbf{x})$ 与增量方向 $\Delta \mathbf{x}$ 之间的夹角。

为了研究函数值如何变化，我们固定增量的步长（即 $\|\Delta \mathbf{x}\|$ 为常数），分析夹角 $\theta$ 对 $\Delta f$ 的影响：

* **下降条件**：只要 $\cos(\theta) < 0$，即 $\theta \in (\frac{\pi}{2}, \frac{3\pi}{2})$，函数值就会下降。
* **下降最快（最速下降）**：当 $\cos(\theta)$ 取最小值时，$\Delta f$ 取得最小（负值绝对值最大）。
  * 当 $\cos(\theta) = -1$ 时，$\Delta f$ 达到最小。
  * 此时 $\theta = \pi$，即 $\Delta \mathbf{x}$ 的方向与梯度方向 $\nabla f(\mathbf{x})$ 完全相反。

**3. 结论与工程实现**

为了让损失函数 $f(\mathbf{x})$ 以最快的速度下降，增量 $\Delta \mathbf{x}$ 应当取负梯度方向。因此，在机器学习的参数更新中，我们定义更新步长为：

$$\Delta \mathbf{x} = -\eta \nabla f(\mathbf{x})$$

其中 $\eta > 0$ 被称为学习率 (Learning Rate)。更新后的参数为：

$$\mathbf{x}_{new} = \mathbf{x}_{old} - \eta \nabla f(\mathbf{x}_{old})$$

---

# 第二部分：感知机 (Perceptron)

**(PDF Page 17 - 25)**

## 1. 理论核心

* **定义**：感知机是神经网络的基石，用于解决**线性可分**问题。
* **模型**：

$$f(x) = \text{sign}(w \cdot x + b)$$

其中 $\text{sign}(z) = +1$ 若 $z \geq 0$，$\text{sign}(z) = -1$ 若 $z < 0$。

* **学习策略**：不是最小化分类错误的个数（因为不可导），而是最小化**误分类点到超平面的总距离**：

$$L(w, b) = -\sum_{x_i \in M} y_i(w \cdot x_i + b)$$

其中 $M$ 为误分类点集合。

* **更新规则**：一旦发现误分类点 $(x_i, y_i)$，就调整 $w$ 和 $b$ 让超平面向该点移动：

$$w \leftarrow w + \eta \, y_i \, x_i$$

$$b \leftarrow b + \eta \, y_i$$

## 2. 辅助工具：感知机迭代逻辑

感知机无法解决非线性问题（如异或 XOR），这是它最大的局限。

## 3. 课后作业 III 解答

**(PDF Page 25)**

**题目**：补全感知机迭代表格。

已知：正样本 $M_1(3,3)$, $M_2(4,3)$；负样本 $M_3(1,1)$。学习率 $\eta = 1$。

当前状态（第4步结束）：$w = (1, 1)^T$，$b = -3$。

**第5步**：选择 $M_1(3,3)$ 作为误分类点。

* **计算验证**：代入 $w \cdot x + b = 1 \times 3 + 1 \times 3 - 3 = 3 > 0$，但需要检查是否所有点分类正确。对 $M_3(1,1)$：$w \cdot x + b = 1 + 1 - 3 = -1 < 0$（正确）。对 $M_1(3,3)$：$3 > 0$（正确）。对 $M_2(4,3)$：$4 + 3 - 3 = 4 > 0$（正确）。若此时所有点分类正确，则算法收敛。

> *注意：由于原始 PDF 中的具体样本坐标和迭代步骤可能与上述示例不同，请以 PDF Page 25 的实际数据为准。感知机迭代的核心逻辑为：找到误分类点 $\rightarrow$ 按更新规则调整 $w, b$ $\rightarrow$ 重复直到无误分类点。*

---

# 第三部分：神经网络与反向传播

**(PDF Page 26 - 57)**

## 1. 理论核心

### 1.1 解决非线性 (Non-linearity)

* **异或问题 (XOR)**：感知机解决不了 XOR 问题，因为找不到一条直线能分开它。
* **多层网络**：通过增加隐藏层，神经网络可以对输入空间进行扭曲、折叠（非线性变换），从而解决复杂分类。

### 1.2 神经网络结构

* **神经元计算**：输入 $x$ $\rightarrow$ 线性变换 ($z = w^T x + b$) $\rightarrow$ 激活函数 ($a = \sigma(z)$)。
* **正向传播**：数据从输入层一层层传到输出层，每一层的计算为：

$$a^{(l+1)} = \sigma\left(W^{(l)} a^{(l)} + b^{(l)}\right)$$

### 1.3 反向传播 (Backpropagation)

* **核心思想**：利用**链式法则**，从输出层开始，将误差 ($\delta$) 一层层向回传递，计算每一层参数的梯度。
* **关键变量 $\delta^{(l)}$**：表示第 $l$ 层的误差项。
  * 输出层误差：

$$\delta^{(L)} = (a^{(L)} - y) \odot \sigma'(z^{(L)})$$

  * 隐藏层误差：

$$\delta^{(l)} = \left( W^{(l+1)T} \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

  * *解释*：每一层的误差 = (上一层传回来的误差权重和) $\odot$ (当前层的激活函数导数)。

* **参数梯度**：

$$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l+1)} \cdot a^{(l)T}, \quad \frac{\partial C}{\partial b^{(l)}} = \delta^{(l+1)}$$

## 2. 辅助代码：手写神经网络 (XOR问题)

这是本章的终极工具，用于理解多层网络如何通过 BP 算法学习。

```python
import numpy as np

# 数据: XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 激活函数
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

# 初始化参数
input_size, hidden_size, output_size = 2, 4, 1
# 对应 PDF 中的 Theta
W1 = np.random.uniform(-1, 1, (input_size, hidden_size)) 
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))

# 训练循环 (BP算法)
lr = 0.5
for i in range(5000):
    # --- 前向传播 (PDF Page 36) ---
    z1 = np.dot(X, W1)
    a1 = sigmoid(z1) # 隐藏层激活值
    z2 = np.dot(a1, W2)
    output = sigmoid(z2) # 输出层预测值
    
    # --- 反向传播 (PDF Page 48-50) ---
    error = y - output
    # 输出层误差 delta^(L)
    d_output = error * sigmoid_deriv(output) 
    # 隐藏层误差 delta^(l) = (delta^(l+1) * W) * f'
    d_hidden = d_output.dot(W2.T) * sigmoid_deriv(a1)
    
    # --- 更新权重 (Gradient Descent) ---
    W2 += a1.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

print("XOR 预测结果:\n", output)

```

## 3. 课后作业 IV & V 解答

**(PDF Page 39, 57)**

**作业 IV (Page 39)**

* **Q1: 为什么多层神经网络可以拟合任意函数？**
  * **A**: 基于**通用近似定理**。只要有非线性激活函数，且隐藏层神经元足够多，神经网络可以通过组合无数个小的非线性变换来逼近任何连续函数。
* **Q2: 深层网络 vs 浅层宽网络？**
  * **A**: 浅层宽网络参数效率低，容易过拟合。深层网络通过层级结构（Layer-wise）提取特征（如边缘 $\rightarrow$ 纹理 $\rightarrow$ 物体），参数利用率更高，泛化能力更强。

**作业 V (Page 57)**

* **Q1: 参数统计 (MNIST案例)**
  * 输入784，隐藏100，输出10。
  * 计算：$(784 \times 100 + 100) + (100 \times 10 + 10) = 78400 + 100 + 1000 + 10 = 79510$ 个参数。
* **Q2: 图片变大 (256x256) 的影响**
  * 输入层变为 $256 \times 256 = 65536$。第一层权重变为 $65536 \times 100 = 655$ 万。参数量剧增，导致计算变慢且极易过拟合。
* **Q3: 降低复杂度的措施**
  * 不使用全连接网络，改用**卷积神经网络 (CNN)**，利用**局部连接**和**权值共享**来大幅减少参数量。

---

# 第四部分：MNIST编程作业解答

**(基于 homework_review.pdf 和 mnist_fcnn_exercise.ipynb)**

## 1. 作业要求

* **任务1**：补全全连接神经网络代码中的6处缺失部分。
* **任务2**：使用sklearn评估模型性能。

## 2. 代码补全解答

**2.1 Sigmoid函数及其导数**

```python
def sigmoid(z):
    """Sigmoid激活函数: σ(z) = 1/(1+e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Sigmoid导数: σ'(z) = σ(z) * (1-σ(z))"""
    return sigmoid(z) * (1.0 - sigmoid(z))
```

**2.2 网络初始化 (__init__)**

```python
# 偏置向量: 对每个隐藏层和输出层生成随机偏置
self._biases = [np.random.randn(y, 1) for y in sizes[1:]]

# 权重矩阵: 连接相邻两层
self._weights = [np.random.randn(y, x)
                 for x, y in zip(sizes[:-1], sizes[1:])]
```

**2.3 前向传播 (feedforward)**

```python
for w, b in zip(self._weights, self._biases):
    a = sigmoid(np.dot(w, a) + b)
```

**2.4 调用反向传播 (update_mini_batch)**

```python
# 对每个样本计算梯度
delta_nabla_b, delta_nabla_w = self.backprop(x, y)
```

**2.5 隐藏层误差计算 (backprop)**

```python
# 误差反向传播: δ^(l) = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^(l))
delta = np.dot(self._weights[-l+1].transpose(), delta) * sp
```

## 3. 完整代码

完整实现见 `mnist_fcnn_solution.py`。

## 4. 性能评估

使用sklearn的classification_report:

```python
from sklearn.metrics import classification_report

# 获取预测结果
predictions = [np.argmax(fc.feedforward(x)) for x, _ in test_data]
labels = [y for _, y in test_data]

# 生成分类报告
print(classification_report(labels, predictions))
```

## 5. 关键点总结

* **权重初始化**：使用标准正态分布，避免对称性问题。
* **矩阵维度**：$W^{(l)}$ 的维度是 (当前层神经元数, 前一层神经元数)。
* **误差传播**：从输出层向输入层逐层计算 $\delta$。
* **梯度累加**：mini-batch 中所有样本的梯度需要累加后再更新。