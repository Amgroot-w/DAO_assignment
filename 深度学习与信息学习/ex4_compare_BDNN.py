
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 日志
tf.logging.set_verbosity(tf.logging.INFO)
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% 0.定义所需函数
def nonlinearity(x):
    return tf.nn.relu(x)

def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - tf.log(tf.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma - (x - mu) ** 2 / (2. * tf.exp(logsigma) ** 2.)

def get_random(shape, avg, std):
    return tf.random_normal(shape, mean=avg, stddev=std)

def log_categ(y, y_hat):
    return tf.reduce_sum(tf.multiply(y, tf.log(y_hat)), axis=1)


# %% 1.读取MNIST数据集
mnist = fetch_openml('mnist_784')
N = 30000
data = np.float32(mnist.data[:]) / 255.  # 数据预处理
idx = np.random.choice(data.shape[0], N)
data = data[idx]
target = np.int32(mnist.target[idx]).reshape(N, 1)  # 标签

train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)  # 划分训练集和测试集
train_data, test_data = data[train_idx], data[test_idx]
train_target0, test_target = target[train_idx], target[test_idx]

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target0))  # 训练集标签one-hot编码


# %% 2.初始化BNN网络中的变量
x = tf.placeholder(tf.float32, shape=None, name='x')  # 输入数据
y = tf.placeholder(tf.float32, shape=None, name='y')  # 输入标签

n_input = train_data.shape[1]  # 训练集维度
M = train_data.shape[0]        # 训练样本数
sigma_prior = 1.0              # 均值sigma的先验
epsilon_prior = 0.001          # 方差epsilon的先验
n_samples = 1                  # n_sample
learning_rate = 0.001          # 学习率
n_epochs = 10                  # 迭代次数
stddev_var = 0.1               # stddev_var

# 2.1 输入层
n_hidden_1 = 200  # 输入层节点数
W1_mu = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=stddev_var))
W1_logsigma = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=0.0, stddev=stddev_var))
b1_mu = tf.Variable(tf.zeros([n_hidden_1]))
b1_logsigma = tf.Variable(tf.zeros([n_hidden_1]))

# 2.2 隐含层
n_hidden_2 = 200  # 隐层节点数
W2_mu = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=stddev_var))
W2_logsigma = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], mean=0.0, stddev=stddev_var))
b2_mu = tf.Variable(tf.zeros([n_hidden_2]))
b2_logsigma = tf.Variable(tf.zeros([n_hidden_2]))

# 2.3 输出层
n_output = 10  # 输出层节点数
W3_mu = tf.Variable(tf.truncated_normal([n_hidden_2, n_output], stddev=stddev_var))
W3_logsigma = tf.Variable(tf.truncated_normal([n_hidden_2, n_output], mean=0.0, stddev=stddev_var))
b3_mu = tf.Variable(tf.zeros([n_output]))
b3_logsigma = tf.Variable(tf.zeros([n_output]))

# 2.4 目标函数
log_pw, log_qw, log_likelihood = 0., 0., 0.


# %% 3.搭建网络结构（绘制TensorFlow计算图）
for _ in range(n_samples):
    # 输入层
    epsilon_w1 = get_random((n_input, n_hidden_1), avg=0., std=epsilon_prior)
    epsilon_b1 = get_random((n_hidden_1,), avg=0., std=epsilon_prior)
    W1 = W1_mu + tf.multiply(tf.log(1. + tf.exp(W1_logsigma)), epsilon_w1)
    b1 = b1_mu + tf.multiply(tf.log(1. + tf.exp(b1_logsigma)), epsilon_b1)
    # 隐层
    epsilon_w2 = get_random((n_hidden_1, n_hidden_2), avg=0., std=epsilon_prior)
    epsilon_b2 = get_random((n_hidden_2,), avg=0., std=epsilon_prior)
    W2 = W2_mu + tf.multiply(tf.log(1. + tf.exp(W2_logsigma)), epsilon_w2)
    b2 = b2_mu + tf.multiply(tf.log(1. + tf.exp(b2_logsigma)), epsilon_b2)
    # 输出层
    epsilon_w3 = get_random((n_hidden_2, n_output), avg=0., std=epsilon_prior)
    epsilon_b3 = get_random((n_output,), avg=0., std=epsilon_prior)
    W3 = W3_mu + tf.multiply(tf.log(1. + tf.exp(W3_logsigma)), epsilon_w3)
    b3 = b3_mu + tf.multiply(tf.log(1. + tf.exp(b3_logsigma)), epsilon_b3)
    # 连接整个网路
    a1 = nonlinearity(tf.matmul(x, W1) + b1)
    a2 = nonlinearity(tf.matmul(a1, W2) + b2)
    h = tf.nn.softmax(nonlinearity(tf.matmul(a2, W3) + b3))

    sample_log_pw, sample_log_qw, sample_log_likelihood = 0., 0., 0.

    for W, b, W_mu, W_logsigma, b_mu, b_logsigma in [(W1, b1, W1_mu, W1_logsigma, b1_mu, b1_logsigma),
                                                     (W2, b2, W2_mu, W2_logsigma, b2_mu, b2_logsigma),
                                                     (W3, b3, W3_mu, W3_logsigma, b3_mu, b3_logsigma)]:
        # first weight prior
        sample_log_pw += tf.reduce_sum(log_gaussian(W, 0., sigma_prior))
        sample_log_pw += tf.reduce_sum(log_gaussian(b, 0., sigma_prior))

        # then approximation
        sample_log_qw += tf.reduce_sum(log_gaussian_logsigma(W, W_mu, W_logsigma * 2))
        sample_log_qw += tf.reduce_sum(log_gaussian_logsigma(b, b_mu, b_logsigma * 2))

    # then the likelihood
    sample_log_likelihood = tf.reduce_sum(log_categ(y, h))

    log_pw += sample_log_pw
    log_qw += sample_log_qw
    log_likelihood += sample_log_likelihood

log_qw /= n_samples
log_pw /= n_samples
log_likelihood /= n_samples

batch_size = 100
n_batches = N / float(batch_size)
n_train_batches = int(train_data.shape[0] / float(batch_size))
minibatch = tf.placeholder(tf.float32, shape=None, name='minibatch')
pi = (1. / n_batches)
objective = tf.reduce_sum(pi * (log_qw - log_pw)) - log_likelihood / float(batch_size)  # 损失函数

optimizer = tf.train.AdamOptimizer(learning_rate)  # 优化器
optimize = optimizer.minimize(objective)  # 优化目标

a1_mu = nonlinearity(tf.matmul(x, W1_mu) + b1_mu)  # 输入层的输出
a2_mu = nonlinearity(tf.matmul(a1_mu, W2_mu) + b2_mu)  # 隐层的输出
h_mu = tf.nn.softmax(nonlinearity(tf.matmul(a2_mu, W3_mu) + b3_mu))  # 输出层的输出
pred = tf.argmax(h_mu, 1)  # 给出预测值

test_ac = []
for k in range(10):
    # %% 4.训练
    sess = tf.Session()  # 开启会话
    sess.run(tf.global_variables_initializer())  # 全局变量初始化

    print('******************** 开始训练 ********************')
    errs = []  # 误差
    test_error = []  # 测试集误差
    for n in range(n_epochs):
        weightVar = []
        for i in range(n_train_batches):
            ob = sess.run([objective, optimize, W2_logsigma, h, y, log_likelihood], feed_dict={
                x: train_data[i * batch_size: (i + 1) * batch_size],
                y: train_target[i * batch_size: (i + 1) * batch_size],
                minibatch: n})
            # 记录训练误差
            errs.append(ob[0])
            weightVar.append(np.mean(ob[2]))
            # 记录测试集误差
            test_pred = sess.run(pred, feed_dict={x: test_data})
            test_error.append(1 - np.count_nonzero(test_pred == np.int32(test_target.ravel())) / float(test_data.shape[0]))

        print('Epoch：%d\tLoss: %.4f \t test error: %.4f' % (n, ob[0], test_error[-1]))
    print('******************** 训练完成 ********************')

    # %% 5.结果
    predictions2 = sess.run(pred, feed_dict={x: test_data})
    ac = np.count_nonzero(predictions2 == np.int32(test_target.ravel())) / float(test_data.shape[0])
    # 记录
    test_ac.append(ac)
    print('第%d次实验，测试集准确率：%.2f' % (k, ac))

pd.DataFrame(test_ac).to_csv(r'Tables\bdnn.csv', index=False)

