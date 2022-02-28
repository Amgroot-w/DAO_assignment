
# %% 导入包，载入数据
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集并展示
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取MNIST数据

# 定义初始化函数，以便重复使用创建权重、偏置、卷积层、池化层。
# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(_x, W):
    return tf.nn.conv2d(_x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool_2x2(_x):
    return tf.nn.max_pool(_x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 构建CNN的结构（绘制tf计算图）
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层。
# 先使用前面函数进行初始化，包括weights和bias。其中[5,5,1,32]代表卷积核尺寸为5**5，1个颜色通道，32个不同的卷积核。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层。
# 基本与第一个卷积层一样，只是其中的卷积核数量变成64.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 定义第一个全连接层。
# 先用reshape将一维的FC层转变为张量，再将其falt处理，展开为一维向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 为了减轻过拟合，使用一个Dropout层，其用法是通过一个placeholder传入keep_prob比率来控制。
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 定义第二个全连接层。
# 输出为10维，并采用softmax输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义损失函数cross_entropy，这里选择Adam优化器。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
optm = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 继续定义评测准确率操作。
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_ac = []
for k in range(10):
    # 训练
    sess = tf.InteractiveSession()  # 创建默认的Interactive Session
    cost = []  # 记录误差变化
    tf.global_variables_initializer().run()  # 全局变量初始化
    # 开始迭代
    for i in range(500):
        # 批训练，每次训练取50张图片
        batch = mnist.train.next_batch(50)
        # 每隔100代展示一次误差
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print(i, train_accuracy)
        # 喂数据，训练
        _, c = sess.run([optm, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # 记录每次迭代的误差，绘制误差变化曲线
        cost.append(c)

    # 测试
    # 全部训练完毕，在最终的测试集上进行全面测试，得到整体的分类准确率。
    ac = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    # 记录
    test_ac.append(ac)
    print('第%d次实验，测试集准确率：%.2f' % (k, ac))

pd.DataFrame(test_ac).to_csv(r'Tables\cnn.csv', index=False)


