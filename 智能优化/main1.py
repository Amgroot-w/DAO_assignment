"""
《智能优化》大作业 --- MOEA/D算法邻域研究
1. 差异的显著性检验

日期：2020.12.25
"""
import numpy as np
import geatpy as ea
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from MyProblem import ZDT1, DTLZ1
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# MOEA/D算法
def moea_d(N=200, epochs=200, trail_num=10):
    # 实例化问题对象
    problem = ZDT1()  # 生成问题对象
    # problem = DTLZ1(3)  # 生成问题对象
    # 种群设置
    Encoding = 'RI'  # 编码方式
    NIND = N  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    # 测试
    trail_num = trail_num  # 重复试验次数
    para_range = [5, 10, 20, 30]  # 参数取值范围
    igd = np.zeros([len(para_range), trail_num])  # 存储igd信息
    for (i, para) in enumerate(para_range):
        for j in range(trail_num):
            myAlgorithm = ea.moea_MOEAD_templet(problem, population)  # 实例化一个MOEA/D算法模板对象
            myAlgorithm.MAXGEN = epochs  # 最大进化代数
            myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
            myAlgorithm.verbose = False  # 设置是否打印输出日志信息
            myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
            myAlgorithm.neighborSize = para  # 设置算法的邻域大小N
            [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
            igd[i, j] = myAlgorithm.log['igd'][-1]  # 记录igd指标
            pd.DataFrame(igd).to_excel('igd.xlsx')  # igd结果保存为文件
            print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
            if myAlgorithm.log is not None and NDSet.sizes != 0:
                print('IGD', myAlgorithm.log['igd'][-1])
                metricName = [['igd'], ['hv']]
                Metrics = np.array([myAlgorithm.log[metricName[i][0]] for i in range(len(metricName))]).T
                # ea.trcplot(Metrics, labels=metricName, titles=metricName)  # 绘制指标追踪分析图
    return igd

# 箱线图
def box_plot(igd):
    trail_num = 10  # 重复试验次数
    para_range = [5, 10, 20, 30]  # 参数取值范围
    data = igd.flatten()
    data_class = np.array([[0]*trail_num, [1]*trail_num, [2]*trail_num, [3]*trail_num]).flatten()
    plot_data = pd.DataFrame({'IGD指标值': data, '参数值': data_class})  # 加上标签构成绘图data
    sns.boxplot(x=data_class, y=data, data=plot_data)
    plt.xlabel('邻域大小')
    plt.xticks(ticks=range(len(igd)), labels=para_range)
    plt.ylabel('IGD指标')
    plt.title('不同邻域大小得到的IGD指标')
    plt.savefig(r'FigsTables\boxplot.png')
    plt.show()

# 差异显著性检验 --- t检验
def ttest(igd):
    igd_mean = np.mean(igd, axis=1)  # 记录igd指标的均值
    pvalue1 = np.zeros([len(igd), len(igd)])
    pvalue = np.zeros([len(igd), len(igd)])
    for i in range(len(igd)):
        for j in range(len(igd)):
            x1 = igd[i, :]
            x2 = igd[j, :]
            # 1.检验是否具有方差齐性
            s1, p1 = stats.levene(x1, x2)
            pvalue1[i, j] = p1
            if p1 < 0.05:
                equal_var = False  # 无方差齐性
            else:
                equal_var = True  # 有方差齐性
            # 2.检验是否有显著性差异
            s, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
            pvalue[i, j] = p
    pd.DataFrame(pvalue1).to_excel(r'FigsTables\pvalue1.xlsx')
    pd.DataFrame(pvalue).to_excel(r'FigsTables\pvalue.xlsx')
    pd.DataFrame(igd_mean).to_excel(r'FigsTables\igd_mean.xlsx')


if __name__ == '__main__':
    start = time.time()
    # MOEA/D
    igd = moea_d(N=200, epochs=200, trail_num=10)
    # 画箱线图
    box_plot(igd)
    # T检验
    ttest(igd)
    end = time.time()
    print('\n用时：%.2f秒' % (end-start))
