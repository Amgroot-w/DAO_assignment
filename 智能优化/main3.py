"""
《智能优化》大作业 --- MOEA/D算法邻域研究
3. 自适应邻域控制策略（策略改编在moea_MOEAD_templet函数中，未完成）

日期：2020.12.25
"""
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
from MyProblem import ZDT1, DTLZ1
# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% 实例化问题对象
problem = ZDT1()  # 生成问题对象
# problem = DTLZ1(3)  # 生成问题对象

# %% 种群设置
Encoding = 'RI'  # 编码方式
NIND = 200  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

for para in [10, 20, 30]:
    # %% 算法参数设置
    myAlgorithm = ea.moea_MOEAD_templet(problem, population)  # 实例化一个MOEA/D算法模板对象
    myAlgorithm.MAXGEN = 200  # 最大进化代数
    myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    myAlgorithm.neighborSize = para
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    # %% 输出结果
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
    if myAlgorithm.log is not None and NDSet.sizes != 0:
        print('IGD', myAlgorithm.log['igd'][-1])




