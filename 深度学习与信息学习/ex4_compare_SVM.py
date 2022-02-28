
import numpy as np
import pandas as pd
from sklearn import svm
from scipy.io import loadmat

# 导入数据
data_x = loadmat('MNIST.mat')['X']
data_y = loadmat('MNIST.mat')['y']
data = np.column_stack((data_x, data_y))

test_ac = []
for i in range(10):
    np.random.shuffle(data)  # 洗牌
    m = data.shape[0]  # 样本总数
    n = data.shape[1]  # 特征数
    images = data[:, :n-1]  # 图片
    labels = data[:, n-1]  # 标签

    # 分配训练集、测试集
    train_num = int(m * 0.7)  # 训练样本数
    test_num = int(m * 0.3)   # 测试样本数
    train_x = images[:train_num, :]
    train_y = labels[:train_num]
    test_x = images[train_num:m, :]
    test_y = labels[train_num:m]

    # 训练
    clf = svm.SVC(C=0.12)
    clf.fit(train_x, train_y)

    # 测试
    ac = clf.score(train_x, train_y)
    ac1 = clf.score(test_x, test_y)

    # 记录
    test_ac.append(ac)
    print('第%d次实验，测试集准确率：%.4f' % (i, ac))

pd.DataFrame(test_ac).to_csv(r'Tables\svm.csv', index=False)













