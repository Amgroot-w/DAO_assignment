### DAO实验室 -《深度学习与信息学习》大作业（代码）

2020.1.7

（若无法运行，请老师查看GitHub仓库地址：https://github.com/Amgroot-w/Deep-learning）

**环境要求：**

+ tensorflow < 2.0 （第一代版本均可）；
+ numpy、pandas、matplotlib、scipy、sklearn 等基础库；
+ seaborn 可视化库。

#### 1. 程序部分

+ ex4_BDNN.py	主程序 --- 直接运行即可得到BDNN模型结果；

+ ex4_cpmpare_BP.py	对比实验程序 --- BP神经网络模型；
+ ex4_cpmpare_SVM.py	对比实验程序 --- 支持向量机模型；
+ ex4_cpmpare_CNN.py	对比实验程序 --- 卷积神经网络模型；
+ ex4_cpmpare.py	对比试验程序 --- 综合以上4种模型的结果；
+ cap.py	外部库函数 --- 实现一些最基础的函数。

#### 2. 数据集部分

MNIST_data、openml、MNIST.mat 均为程序所需数据集，需要放在根目录下程序才能正常运行。

#### 3. 图表部分

+ Figs 目录下为程序生成的图像文件；
+ Tables 目录下为程序生成的表格文件。