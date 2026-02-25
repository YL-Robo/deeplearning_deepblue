## 引入python包，没有安装的请按照抛出的error通过conda来安装直至成功
import numpy as np

# torchvision 是 PyTorch 官方配套的计算机视觉（Computer Vision）工具包，专门为深度学习中的视觉任务打造
# 它和 PyTorch 深度集成，能帮你省去大量重复的基础开发工作，是计算机视觉领域最常用的工具之一。
import torchvision

#  NumPy 库的打印配置函数，让 NumPy 打印数组时，显示数组里的所有元素，不再用省略号 ... 隐藏部分内容
np.set_printoptions(threshold=np.inf)

def onehot(targets, num):
    """将数字的label转换成One-Hot的形式"""
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


def img2col(x, ksize, step):
    """
    将图像中所有需要卷积的地方转化成矩阵，方便卷积加速
    :param x: 图像
    :param ksize: 卷积大小
    :param step: 步长
    :return: 二维矩阵，每一行是所有深度上待卷积部分的一维形式
    """
    # [width,height,channel] 宽，长，深度
    wx, hx, cx = x.shape
    # 返回的特征图尺寸
    feature_w = (wx - ksize) // step + 1
    image_col = np.zeros((feature_w * feature_w, ksize * ksize * cx))
    num = 0
    ## 补全代码，补充image_col具体数值 ##
    for i in range(feature_w):
        for j in range(feature_w):
            image_col[num] = x[i*step:i*step+ksize, j*step:j*step+ksize, :].reshape(-1)
            num += 1
    return image_col


## Relu 函数
class Relu(object):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


## Softmax 函数
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

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i])
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax


dataset_path = "./dataset/mnist"
train_data = torchvision.datasets.MNIST(
            root=dataset_path,  # 参数1：数据存储路径
            train=True,         # 参数2：指定加载训练集/测试集(True:加载训练集 False:加载测试集)
            download=False      # 参数3：是否自动下载数据集
)
train_data.data = train_data.data.numpy()  # [60000,28,28]
train_data.targets = train_data.targets.numpy()  # [60000]
train_data.data = train_data.data.reshape(60000, 28, 28, 1) / 255.  # 输入向量处理
train_data.targets = onehot(train_data.targets, 60000)  # 标签one-hot处理 (60000, 10)


## 全连接层
class Linear(object):

    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel / 2)
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        """前向过程"""
        ## 补全代码 ##

    def backward(self, delta, learning_rate):
        """反向过程"""
        ## 梯度计算
        batch_size = self.x.shape[0]

        ## 补全代码 ##
        self.W_gradient =
        self.b_gradient =
        delta_backward =
        ## 反向传播
        self.W -=
        self.b -=

        return delta_backward


## conv
class Conv(object):
    def __init__(self, kernel_shape, step=1, pad=0):
        # [w, h, d]
        width, height, in_channel, out_channel = kernel_shape
        self.step = step
        self.pad = pad
        scale = np.sqrt(3 * in_channel * width * height / out_channel)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(out_channel) / scale
        self.k_gradient = np.zeros(kernel_shape)
        self.b_gradient = np.zeros(out_channel)

    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')
        bx, wx, hx, cx = self.x.shape
        # kernel的宽、高、通道数、个数
        wk, hk, ck, nk = self.k.shape
        feature_w = (wx - wk) // self.step + 1  # 返回的特征图尺寸
        feature = np.zeros((bx, feature_w, feature_w, nk))

        self.image_col = []
        # kernal也进行了reshape，便于卷积加速，只保留通道维度，是个二维的矩阵
        kernel = self.k.reshape(-1, nk)

        ## 补全代码 ##
        for i in range(bx):
            image_col =
            feature[i] =
            self.image_col.append()
        return feature

    def backward(self, delta, learning_rate):
        bx, wx, hx, cx = self.x.shape  # batch,14,14,inchannel
        wk, hk, ck, nk = self.k.shape  # 5,5,inChannel,outChannel
        bd, wd, hd, cd = delta.shape  # batch,10,10,outChannel

        # 计算self.k_gradient,self.b_gradient
        # 参数更新过程
        ## 补全代码 ##
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient +=
        self.k_gradient /=
        self.b_gradient +=
        self.b_gradient /=

        # 计算delta_backward
        # 误差的反向传递
        delta_backward = np.zeros(self.x.shape)
        # numpy矩阵（对应kernal）旋转180度
        ## 补全代码 ##
        k_180 =
        k_180 =
        k_180_col =

        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = img2col(pad_delta[i], wk, self.step)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx, hx, ck)

        # 反向传播
        self.k -= self.k_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


## Max Pooling层
class Pool(object):
    def forward(self, x):
        b, w, h, c = x.shape
        feature_w = w // 2
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))  # 记录最大池化时最大值的位置信息用于反向传播
        for bi in range(b):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        ## 补全代码
                        feature[bi, i, j, ci] =
                        index = np.argmax(x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        self.feature_mask[bi, i * 2 + index // 2, j * 2 + index % 2, ci] = 1
        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


def train(batch=32, lr=0.01, epochs=10):
    # Mnist手写数字集
    dataset_path = "./dataset/mnist"
    train_data = torchvision.datasets.MNIST(root=dataset_path, train=True, download=False)
    train_data.data = train_data.data.numpy()  # [60000,28,28]
    train_data.targets = train_data.targets.numpy()  # [60000]
    train_data.data = train_data.data.reshape(60000, 28, 28, 1) / 255.  # 输入向量处理
    train_data.targets = onehot(train_data.targets, 60000)  # 标签one-hot处理 (60000, 10)

    # [28,28] 卷积 6x[5,5] -> 6x[24,24]
    conv1 = Conv(kernel_shape=(5, 5, 1, 6))
    relu1 = Relu()
    # 6x[24,24] -> 6x[12,12]
    pool1 = Pool()
    # 6x[12,12] 卷积 16x(6x[12,12]) -> 16x[8,8]
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    # 16x[8,8] -> 16x[4,4]
    pool2 = Pool()

    # 在这里可以尝试增加网络的深度，再实例化conv3和pool3，记得后面的前向传播过程
    # 和反向传播过程也要有对应的过程

    nn = Linear(256, 10)
    softmax = Softmax()

    for epoch in range(epochs):
        for i in range(0, 60000, batch):
            X = train_data.data[i:i + batch]
            Y = train_data.targets[i:i + batch]

            # 前向传播过程
            predict = conv1.forward(X)
            predict = relu1.forward(predict)
            predict = pool1.forward(predict)
            predict = conv2.forward(predict)
            predict = relu2.forward(predict)
            predict = pool2.forward(predict)
            predict = predict.reshape(batch, -1)
            predict = nn.forward(predict)

            # 误差计算
            loss, delta = softmax.cal_loss(predict, Y)

            # 反向传播过程
            delta = nn.backward(delta, lr)
            delta = delta.reshape(batch, 4, 4, 16)
            delta = pool2.backward(delta)
            delta = relu2.backward(delta)
            delta = conv2.backward(delta, lr)
            delta = pool1.backward(delta)
            delta = relu1.backward(delta)
            conv1.backward(delta, lr)

            print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        lr *= 0.95 ** (epoch + 1)
        np.savez("simple_cnn_model.npz", k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=nn.W, b3=nn.b)



def eval():
    model = np.load("simple_cnn_model.npz")

    dataset_path = "./dataset/mnist"
    test_data = torchvision.datasets.MNIST(root=dataset_path, train=False)
    test_data.data = test_data.data.numpy()  # [10000,28,28]
    test_data.targets = test_data.targets.numpy()  # [10000]

    test_data.data = test_data.data.reshape(10000, 28, 28, 1) / 255.

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))  # 24x24x6
    relu1 = Relu()
    pool1 = Pool()  # 12x12x6
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16
    nn = Linear(256, 10)
    softmax = Softmax()

    conv1.k = model["k1"]
    conv1.b = model["b1"]
    conv2.k = model["k2"]
    conv2.b = model["b2"]
    nn.W = model["w3"]
    nn.n = model["b3"]

    num = 0
    for i in range(10000):
        X = test_data.data[i]
        X = X[np.newaxis, :]
        Y = test_data.targets[i]

        predict = conv1.forward(X)
        predict = relu1.forward(predict)
        predict = pool1.forward(predict)
        predict = conv2.forward(predict)
        predict = relu2.forward(predict)
        predict = pool2.forward(predict)
        predict = predict.reshape(1, -1)
        predict = nn.forward(predict)

        predict = softmax.predict(predict)

        if np.argmax(predict) == Y:
            num += 1

    print("TEST-ACC: ", num / 10000 * 100, "%")


if __name__ == "__main__":
    train()
    eval()