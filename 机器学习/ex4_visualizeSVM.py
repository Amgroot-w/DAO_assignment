"""
《机器学习》作业 --- SVM结果展示

时间：2021.1.15

"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, 0])

plt.figure()
plt.scatter(x[:, 0], x[:, 1], s=100, marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(0, 5)
plt.ylim(0, 4)
plt.show()

plt.figure()
plt.scatter(x[:2, 0], x[:2, 1], c='blue', s=100, marker='o')
plt.scatter(x[2, 0], x[2, 1], c='red', s=120, marker='X')
plt.plot([0.5, 3.5], [3.75, 0.25], '--',)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(0, 4.5)
plt.ylim(0, 4)
plt.show()















