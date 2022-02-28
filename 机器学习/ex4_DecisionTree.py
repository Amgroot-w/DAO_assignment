"""
《机器学习》作业 --- 决策树

时间：2021.1.13

"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import matplotlib.pyplot as plt

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# pandas 读取 csv 文件
data = pd.read_csv('data.csv', header=None)
data.columns = ['temperature', 'snot', 'muscle', 'headache', 'cold']  # 指定列

# sparse=False 不产生稀疏矩阵
vec_x = DictVectorizer(sparse=False)
vec_y = DictVectorizer(sparse=False)
# 先用 pandas 对每行生成字典，然后进行向量化
feature = data[['temperature', 'snot', 'muscle', 'headache']]
label = data[['cold']]
# 划分训练集、测试集
x = vec_x.fit_transform(feature.to_dict(orient='record'))
y = vec_y.fit_transform(label.to_dict(orient='record'))[:, 1]
train_num = 16
x_train = x[:train_num, :]
y_train = y[:train_num]
x_test = x[train_num:, :]
y_test = y[train_num:]

# 打印各个变量
print('\n特征\n', feature)
print('\n转化后的矩阵\n', x_train)
print('\n矩阵的列名\n', vec_x.get_feature_names())

# 训练决策树
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)

# 可视化决策树
feature_names = ['头痛', '肌肉痛', '流鼻涕', '体温=High', '体温=Normal', '体温=Very High']
plt.figure(figsize=[10, 10])
tree.plot_tree(clf, feature_names=feature_names, filled=True, rounded=True, fontsize=12)
plt.savefig(r'Figs\tree.png')
plt.show()



