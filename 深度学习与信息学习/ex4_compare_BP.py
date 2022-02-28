
import numpy as np
import pandas as pd
from scipy.io import loadmat
import cap

# 导入数据
images = loadmat('MNIST.mat')['X']  # 图片
labels = loadmat('MNIST.mat')['y']  # 标签
m = images.shape[0]  # 样本总数
n = images.shape[1]  # 特征数

test_ac = []
for i in range(10):
    # 打乱数据
    data = np.column_stack((images, labels))
    np.random.shuffle(data)  # 洗牌
    images = data[:, :n]
    labels = data[:, n].reshape([m, 1])

    # 分配训练集、测试集
    train_num = int(m * 0.7)  # 训练样本数
    test_num = int(m * 0.3)   # 测试样本数
    train_x = images[:train_num, :]
    train_y = labels[:train_num, :]
    test_x = images[train_num:m, :]
    test_y = labels[train_num:m, :]

    # 归一化处理
    train_x = cap.normalize(train_x, 'maxmin')  # 特征归一化处理
    test_x = cap.normalize(test_x, 'maxmin')

    # 异常值处理
    train_x = pd.DataFrame(train_x).fillna(0).values  # nan替换为0
    test_x = pd.DataFrame(test_x).fillna(0).values  # nan替换为0

    # 标签采用one-hot编码
    train_y = cap.onehot(train_y - 1)
    test_y = cap.onehot(test_y - 1)

    # 搭建神经网络框架
    input_num = 400  # 输入节点数
    hidden_num = 25  # 隐层节点数
    output_num = 10  # 输出节点数

    # 初始化权重
    threshold = np.sqrt(6) / np.sqrt(400+25)
    w1 = np.random.uniform(-threshold, threshold, [input_num, hidden_num])  # 公式初始化方法
    threshold = np.sqrt(6) / np.sqrt(25+10)
    w2 = np.random.uniform(-threshold, threshold, [hidden_num, output_num])
    b1 = np.zeros(hidden_num)
    b2 = np.zeros(output_num)

    # 设置超参数
    alpha = 1  # 学习率
    lamda = 0  # 正则化参数
    epochs = 2000  # 迭代次数
    # 训练
    cost = []
    for epoch in range(epochs):
        # 前向传播
        hidden_in = np.matmul(train_x, w1) + b1
        hidden_out = cap.sigmoid(hidden_in)
        network_in = np.matmul(hidden_out, w2) + b2
        network_out = cap.softmax(network_in)
        # 记录总误差
        J = cap.cross_entropy(network_out, train_y) \
            + 1/(2*m) * lamda * (np.sum(w2**2)+np.sum(w1**2))
        cost.append(J)
        # 反向传播
        output_delta = network_out - train_y
        hidden_delta = np.multiply(np.matmul(output_delta, w2.T), np.multiply(hidden_out, 1-hidden_out))
        # 梯度更新
        dw2 = 1/train_num * (np.matmul(hidden_out.T, output_delta) + lamda*w2)
        db2 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, output_delta)
        dw1 = 1/train_num * (np.matmul(train_x.T, hidden_delta) + lamda*w1)
        db1 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, hidden_delta)
        w2 = w2 - alpha*dw2
        w1 = w1 - alpha*dw1
        b2 = b2 - alpha*db2
        b1 = b1 - alpha*db1

    # 测试集准确率
    hidden_in = np.matmul(test_x, w1) + b1
    hidden_out = cap.sigmoid(hidden_in)
    network_in = np.matmul(hidden_out, w2) + b2
    network_out = cap.sigmoid(network_in)
    test_pred = np.argmax(network_out, axis=1)
    test_y = np.argmax(test_y, axis=1)
    ac = float(np.mean(np.equal(test_pred, test_y)))
    # 记录
    test_ac.append(ac)
    print('第%d次实验，测试集准确率：%.2f' % (i, ac))

pd.DataFrame(test_ac).to_csv(r'Tables\bp.csv', index=False)



