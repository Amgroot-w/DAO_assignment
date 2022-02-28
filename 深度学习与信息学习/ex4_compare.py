
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读入数据
svm = pd.read_csv(r'Tables\svm.csv')
bp = pd.read_csv(r'Tables\bp.csv')
cnn = pd.read_csv(r'Tables\cnn.csv')
bdnn = pd.read_csv(r'Tables\bdnn.csv')

# 箱型图
data = pd.concat((svm, bp, cnn, bdnn), axis=0).values.reshape(-1)
name = np.array([['SVM']*10, ['BP']*10, ['CNN']*10, ['BDNN']*10]).flatten()
plot_data = pd.DataFrame({'准确率': data, '模型': name})
sns.boxplot(x=name, y=data, data=plot_data)
plt.xlabel('模型')
plt.xticks(ticks=range(4), labels=['SVM', 'BP', 'CNN', 'BDNN'], fontsize=12)
plt.ylabel('准确率')
plt.title('4类模型10次迭代的准确率')
plt.savefig(r'Figs\ex4_4类模型10次迭代的准确率')
plt.show()

# 求均值和方差
ac = pd.concat((svm, bp, cnn, bdnn), axis=1).values.T
ac_mean = np.mean(ac, axis=1)  # 均值
ac_std = np.std(ac, axis=1)  # 标准差
ac_table = np.row_stack((ac.T, ac_mean, ac_std))
pd.DataFrame(ac_table).to_excel(r'Tables\ex4_ac.xlsx')


# 差异的显著性检验
pvalue1 = np.zeros([len(ac), len(ac)])
pvalue = np.zeros([len(ac), len(ac)])
for i in range(len(ac)):
    for j in range(len(ac)):
        x1 = ac[i, :]
        x2 = ac[j, :]
        # 1.检验是否具有方差齐性
        s1, p1 = stats.levene(x1, x2)
        pvalue1[i, j] = p1
        if p1 < 0.05:
            equal_var = False  # 无有方差齐性
        else:
            equal_var = True  # 有方差齐性
        # 2.检验是否有显著性差异
        s, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
        pvalue[i, j] = p

pd.DataFrame(pvalue1).to_excel(r'Tables\ex4_pvalue1.xlsx')
pd.DataFrame(pvalue).to_excel(r'Tables\ex4_pvalue.xlsx')




