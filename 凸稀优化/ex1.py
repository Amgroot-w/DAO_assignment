"""
凸稀优化作业1-SGD
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

def f(x1, x2):
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)


# 展示原函数的等值线图
plt.figure()
xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
zz = f(xx, yy)
plt.contour(xx, yy, zz)
plt.show()

x1_init = np.random.random()  # 初始化x1
x2_init = np.random.random()  # 初始化x2
alpha = 0.0007  # 学习率
epochs = 1000  # 迭代次数

# 1. Cyclic Rule
x1 = copy.deepcopy(x1_init)
x2 = copy.deepcopy(x2_init)
f_log1 = []
while len(f_log1) < epochs:
    # 更新x1
    dx1 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1)
    x1 = x1 - alpha * dx1
    ff = f(x1, x2)
    if len(f_log1) % 100 == 0:
        print('epoch: ', len(f_log1), 'f:', ff)
    f_log1.append(ff)

    # 更新x2
    dx2 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    x2 = x2 - alpha * dx2
    ff = f(x1, x2)
    if len(f_log1) % 100 == 0:
        print('epoch: ', len(f_log1), 'f:', ff)
    f_log1.append(ff)
print('目标函数f(x1,x2)的最小值：%.4f, 最优决策变量(x1,x2)：(%4f,%.4f)\n' % (ff, x1, x2))

# 2. Randomized Rule
x1 = copy.deepcopy(x1_init)
x2 = copy.deepcopy(x2_init)
f_log2 = []
for epoch in range(epochs):
    # 随机选择
    if np.random.random() < 0.5:
        dx1 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1)
        x1 = x1 - alpha * dx1
        ff = f(x1, x2)
        if epoch % 100 == 0:
            print('epoch: ', epoch, 'f:', ff)
        f_log2.append(ff)

    else:
        dx2 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
        x2 = x2 - alpha * dx2
        ff = f(x1, x2)
        if epoch % 100 == 0:
            print('epoch: ', epoch, 'f:', ff)
        f_log2.append(ff)
print('目标函数f(x1,x2)的最小值：%.4f, 最优决策变量(x1,x2)：(%.4f,%.4f)\n' % (ff, x1, x2))

# 展示两种方法求解迭代过程图
plt.figure()
plt.plot(range(len(f_log1)), f_log1, label='Cyclic Rule')
plt.plot(range(len(f_log2)), f_log2, label='Randomized  Rule')
plt.xlabel('epoch')
plt.ylabel('f(x1,x2)')
plt.legend()
plt.show()












