"""
2021.4.28

"""
# 忽略警告
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import xgboost as xgb
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 分割25个station数据
def station_split(dataset):
    station_split_data = list()
    for i in range(25):
        station_split_data.append(dataset[dataset['station'] == i + 1])
    return station_split_data


# 缺失值处理
def fillna(dataset, method='linear'):
    """
    :param dataset: 要进行缺失值处理的数据集（DataFrame类型）
    :param method: 缺失值填补方法（默认：线性差值）
    :return 缺失值填补之后的数据集（DataFrame类型）
    """
    return dataset.interpolate(method=method)


# 展示指定station的缺失数据分布情况
def show_station_data_nan(dataset, station_index):
    """
    :param dataset: 要展示的数据集（按照25个station分好之后的list类型）
    :param station_index: 选择要展示第几个station（序号：1~25）
    """
    heat_map = pd.isna(dataset[station_index].iloc[:, 2:-2])
    sns.heatmap(heat_map, cbar=False)
    yticks = np.append(np.arange(1, 311, 31), 309)
    ylabels = dataset[station_index]['Date'].iloc[yticks]
    plt.xticks(rotation=70)
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.title('Station %d' % (station_index + 1))
    plt.show()


# 展示缺失值填充结果（指定station、指定feature）
def show_station_data_fillna(dataset, nan_index, station_index, feature_index):
    """
    :param dataset: 缺失值处理后的数据集（按照25个station分好之后的list类型）
    :param nan_index: 缺失值的index矩阵（size与dataset相同）
    :param station_index: 选择要展示第几个station（序号：0~24， 共25个station）
    :param feature_index: 选择展示第几个feature（序号：0~20，共21个feature）
    """
    plot_x = np.arange(310)
    plot_y = dataset[station_index].iloc[:, feature_index + 2]  # 选择要展示的station和feature对应的列向量数据
    isna_index = nan_index[station_index].iloc[:, feature_index + 2]  # 选择要展示的station和feature对应的缺失值index
    legend = ['实际值' if not i else '填充值' for i in isna_index]  # 将布尔型的index转化为字符型legend
    plot_data = {'x': plot_x, 'y': plot_y, 'legend': legend}  # 组装成seaborn的绘图data

    plt.figure(figsize=(20, 5))
    sns.scatterplot(data=plot_data, x='x', y='y', hue='legend', s=50, alpha=0.75)
    plt.plot(plot_x, plot_y, '-', linewidth=0.5)
    xticks = np.append(np.arange(1, 311, 31), 309)
    xlabels = dataset[station_index + 1]['Date'].iloc[xticks]
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.title('Station %d' % (station_index + 1))
    plt.xlabel('Date')
    plt.show()


# 异常值：箱线图可视化
def box_plot(dataset):
    plt.figure(figsize=(30, 10))  # 第1~8个特征
    for i in np.arange(2, 10):
        y_name = dataset.columns[i]
        plt.subplot(2, 4, i - 1)
        sns.boxplot(x='station', y=y_name, data=dataset, palette='Set3', whis=2)
    plt.show()

    plt.figure(figsize=(30, 10))  # 第9~16个特征
    for i in np.arange(10, 18):
        y_name = dataset.columns[i]
        plt.subplot(2, 4, i - 9)
        sns.boxplot(x='station', y=y_name, data=dataset, palette='Set3', whis=2)
    plt.show()

    plt.figure(figsize=(30, 10))  # 第17~21个特征 + 目标值Next_Tmax
    for i in np.arange(18, 24):
        y_name = dataset.columns[i]
        plt.subplot(2, 3, i - 17)
        sns.boxplot(x='station', y=y_name, data=dataset, palette='Set3', whis=2)
    plt.show()


# 根据Date划分数据集（hindcast validation）
def train_val_split_by_Date(dataset, train_years, val_years):
    """
    :param dataset: 原始数据集(归一化后)，DataFrame格式
    :param train_years: 训练集的年份（列表，0表示第一年，即2013年）
    :param val_years: 验证集的年份（列表，0表示第一年，即2013年）
    :return: 训练集特征值、训练集目标值、验证集特征值、验证集目标值
    """
    # 将dataset按年份分为5份，得到5*1550*25矩阵
    dataset = dataset.values.reshape(5, -1, dataset.shape[1])
    # 提取训练集、验证集数据，并去掉station序号、Date、Next_Tmin，得到m*22矩阵
    train_x = dataset[train_years].reshape(-1, dataset.shape[2])[:, 2:23]
    train_y = dataset[train_years].reshape(-1, dataset.shape[2])[:, 23]
    val_x = dataset[val_years].reshape(-1, dataset.shape[2])[:, 2:23]
    val_y = dataset[val_years].reshape(-1, dataset.shape[2])[:, 23]
    return train_x, train_y, val_x, val_y


# 根据station划分数据集（Leave-one-station-out Validation, LOSOV）
def train_val_split_by_station(dataset, station_index):
    """
    :param dataset: 归一化后的数据集，list类型，每个元素为一个DataFrame
    :param station_index: 作为验证集的station序号
    :return: 训练集特征值、训练集目标值、验证集特征值、验证集目标值
    """
    train_x = []
    train_y = []
    for i in range(25):
        if i == station_index:
            val_x = dataset[i].iloc[:, 2:23]
            val_y = dataset[i].iloc[:, 23]
        else:
            train_x.append(dataset[i].iloc[:, 2:23])
            train_y.append(dataset[i].iloc[:, 23])
    train_x = np.array(train_x).reshape(-1, 21)
    train_y = np.array(train_y).reshape(-1, 1)
    return train_x, train_y, val_x, val_y


# 归一化（maxmin方法）
def normalize(dataset):
    # 将输入的dataset转化为二维np数组
    dataset = np.array(dataset).reshape(len(dataset), -1)
    # 按列（特征）进行归一化
    scaler = MinMaxScaler()
    res = scaler.fit_transform(dataset)
    return res, scaler


# 打印指标值
def print_metrics(model_name, R_square, RMSE):
    print('%s模型: R_square: %.4f\tRMSE: %.4f' % (model_name, R_square, RMSE))


# 展示温度预测效果
def plot_prediction_and_true(model_name, pred, true):
    plt.figure(figsize=(50, 10))
    plt.plot(range(len(pred)), true, 'o-', linewidth=4, label='真实值')
    plt.plot(range(len(pred)), pred, 'o-', linewidth=4, label='预测值')
    plt.xlabel('样本序数', fontsize=35)
    plt.ylabel('Next_Tmax', fontsize=35)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('%s模型的预测效果' % model_name, fontsize=35)
    plt.legend(fontsize=35)
    plt.show()


# 打印losov结果
def losov_print(losov_info):
    # 计算25个station的2个指标的平均值
    MetricsAvg = np.mean(np.array(losov_info['metrics']), axis=0)
    for (i, model_name) in enumerate([' RF', 'SVR', 'ANN', 'MME']):
        # 打印4个模型的2个指标
        print('%s模型:   训练集：R_square: %.4f, RMSE: %.4f;   验证集：R_square: %.4f, RMSE: %.4f'
              % (model_name, MetricsAvg[i, 0], MetricsAvg[i, 1], MetricsAvg[i, 2], MetricsAvg[i, 3]))

# 打印losov结果
def losov_print1(losov_info):
    # 计算25个station的2个指标的平均值
    MetricsAvg = np.mean(np.array(losov_info['metrics']), axis=0)
    for (i, model_name) in enumerate(['LR', 'Ridge', '///', '///']):
        # 打印4个模型的2个指标
        print('%s模型:   训练集：R_square: %.4f, RMSE: %.4f;   验证集：R_square: %.4f, RMSE: %.4f'
              % (model_name, MetricsAvg[i, 0], MetricsAvg[i, 1], MetricsAvg[i, 2], MetricsAvg[i, 3]))


# 绘制station图
def losov_plot(losov_info):
    # 从训练数据中提取出seaborn绘图数据
    plot_data = {
        'R_square': np.array(losov_info['metrics']).reshape(-1, 4)[:, 2],
        'RMSE': np.array(losov_info['metrics']).reshape(-1, 4)[:, 3],
        'model_name': np.array(losov_info['model_names']).reshape(-1, ),
        'station_index': np.array(losov_info['station_index']).reshape(-1, )
    }
    # 设置画图参数
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)
    # 绘制R_square图
    plt.figure(figsize=(25, 5))
    sns.barplot(x='station_index', y='R_square', data=plot_data, hue='model_name')
    plt.xlabel('Station')
    plt.ylabel('R_square')
    plt.title('R^2 in 25 stations')
    plt.legend(loc='lower right')
    plt.show()
    # 绘制RMSE图
    plt.figure(figsize=(25, 5))
    sns.barplot(x='station_index', y='RMSE', data=plot_data, hue='model_name')
    plt.xlabel('Station')
    plt.ylabel('RMSE')
    plt.title('RMSE in 25 stations')
    plt.legend(loc='upper right')
    plt.show()


# 输出评价指标得分
def get_metrics(model_name, true, pred, y_scaler, print_=False, plot_=False):
    # 将真实值、预测值反归一化
    true_scale = y_scaler.inverse_transform(true)
    pred_scale = y_scaler.inverse_transform(pred.reshape(-1, 1))
    # 计算评价指标值
    R_square = r2_score(true_scale, pred_scale)  # 决定系数
    RMSE = mean_squared_error(true_scale, pred_scale, squared=False)  # 均方根误差
    if print_:
        print_metrics(model_name, R_square, RMSE)  # 输出指标值
    if plot_:
        plot_prediction_and_true(model_name, pred_scale, true_scale)  # 展示拟合效果
    return R_square, RMSE

# 偏差修正模型类（单模型）
class bias_correction_model(object):
    """
    单模型：
       随机森林；
       支持向量回归；
       人工神经网络；
    """

    def __init__(self, method):
        self.model_name = method  # 记录模型名称
        if method == 'RF':
            self.RF_model_init()  # 随机森林
        elif method == 'SVR':
            self.SVR_model_init()  # 支持向量回归
        elif method == 'ANN':
            self.ANN_model_init()  # 人工神经网络

    # 初始化：随机森林RF
    def RF_model_init(self):
        # self.model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1)
        self.model = xgb.XGBRegressor(n_estimators=100)

    # 初始化：支持向量回归SVR
    def SVR_model_init(self):
        self.model = SVR(kernel='rbf', gamma=0.06, C=0.055)

    # 初始化：人工神经网络ANN
    def ANN_model_init(self):
        self.model = MLPRegressor(hidden_layer_sizes=10, activation='relu', solver='adam',
                                  alpha=0.001, batch_size='auto', learning_rate='adaptive',
                                  learning_rate_init=0.1, max_iter=1000)

    # 训练模型
    def fit(self, x, y):
        self.model.fit(x, y)

    # 预测输出
    def predict(self, x):
        # 注意：此函数输出的pred是（0,1）区间上的值（目标值y归一化的结果）
        self.pred = self.model.predict(x)
        # 将预测输出限制为二维np数组
        self.pred = self.pred.reshape(-1, 1)
        return self.pred

    # 输出评价指标得分
    def score(self, x, y, y_scaler, print_=True, plot_=True):
        # 得到预测值
        self.predict(x)

        # 将真实值、预测值反归一化
        y = y_scaler.inverse_transform(y)
        self.pred = y_scaler.inverse_transform(self.pred)

        # 计算评价指标值
        R_square = r2_score(y, self.pred)  # 决定系数
        RMSE = mean_squared_error(y, self.pred, squared=False)  # 均方根误差

        if print_:
            print_metrics(self.model_name, R_square, RMSE)  # 输出指标值
        if plot_:
            plot_prediction_and_true(self.model_name, self.pred, y)  # 展示拟合效果

        return R_square, RMSE


# 偏差修正模型类（集成模型）
class bias_correction_ensemble_model(object):
    """
    集成方法：
       简单加权，为3个子模型分别赋予一个权重，然后训练；
       权重的范围不限制在0~1之间，也不限制权重之和必须为1；
       权重初始化采用标准正态分布。
    """

    def __init__(self, epochs=2000, learning_rate=0.001):
        """
        :param epochs: 迭代次数
        :param learning_rate: 学习率
        """
        self.epochs = epochs  # 迭代次数
        self.learning_rate = learning_rate  # 学习率

        self.models = [
            bias_correction_model(method='RF'),
            bias_correction_model(method='SVR'),
            bias_correction_model(method='ANN')
        ]
        self.model_num = len(self.models)  # 集成模型的子模型个数
        self.model_name = 'MME'

    # 绘制计算图
    def draw_computation_map(self):
        # 集成模型的三个输入
        self.x1 = tf.placeholder(tf.float32, shape=(None, 1))
        self.x2 = tf.placeholder(tf.float32, shape=(None, 1))
        self.x3 = tf.placeholder(tf.float32, shape=(None, 1))
        # 各子模型的权重
        self.w1 = tf.Variable(initial_value=np.random.randn())
        self.w2 = tf.Variable(initial_value=np.random.randn())
        self.w3 = tf.Variable(initial_value=np.random.randn())
        # 真实值
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        # 预测值
        self.y_hat = self.w1 * self.x1 + self.w2 * self.x2 + self.w3 * self.x3
        # 损失函数
        self.cost = 1 / 2 * tf.reduce_mean((self.y_hat - self.y) ** 2)  # MSE
        # 优化器
        self.optm = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    # 训练模型
    def fit(self, x, y, show_train_process=True, print_w=True):
        preds = []
        for i in range(self.model_num):
            self.models[i].fit(x, y)  # 依次训练子模型
            preds.append(self.models[i].predict(x))  # 得到各个子模型的输出

        # 训练集成模型（集成模型各自的权重）
        self.draw_computation_map()  # 搭建计算图
        self.sess = tf.Session()  # 会话
        cost_log = []
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.epochs):
            _, c = self.sess.run([self.optm, self.cost], feed_dict={
                self.x1: preds[0], self.x2: preds[1], self.x3: preds[2], self.y: y})
            cost_log.append(c)

        if show_train_process:
            plt.plot(range(len(cost_log)), cost_log)
            plt.title('加权集成模型：训练误差变化曲线')
            plt.show()
        if print_w:
            print('集成模型的权重：w1=%.4f, w2=%.4f, w3=%.4f'
                  % (self.w1.eval(self.sess), self.w2.eval(self.sess), self.w3.eval(self.sess)))

    # 预测输出
    def predict(self, x):
        preds = []
        for i in range(self.model_num):
            preds.append(self.models[i].predict(x))
        pred = self.sess.run(self.y_hat, feed_dict={self.x1: preds[0], self.x2: preds[1], self.x3: preds[2]})
        return pred

    # 输出评价指标得分
    def score(self, x, y, y_scaler, print_=True, plot_=True):
        # # 展示子模型的效果
        # self.models[0].score(x, y, y_scaler, print_=True, plot_=False)
        # self.models[1].score(x, y, y_scaler, print_=True, plot_=False)
        # self.models[2].score(x, y, y_scaler, print_=True, plot_=False)

        # 得到预测值
        self.pred = self.predict(x)

        # 将真实值、预测值反归一化
        y_scale = y_scaler.inverse_transform(y)
        self.pred_scale = y_scaler.inverse_transform(self.pred)

        # 计算评价指标值
        R_square = r2_score(y_scale, self.pred_scale)  # 决定系数
        RMSE = mean_squared_error(y_scale, self.pred_scale, squared=False)  # 均方根误差

        if print_:
            print_metrics(self.model_name, R_square, RMSE)  # 输出指标值
        if plot_:
            plot_prediction_and_true(self.model_name, self.pred, y)  # 展示拟合效果

        return R_square, RMSE


# LSTM模型
class lstm(object):
    # 初始化
    def __init__(self, input_num, hidden_num, output_num,
                 time_step, batch_size, learning_rate, lamda, epochs):
        self.input_num = input_num          # 输入节点数
        self.hidden_num = hidden_num        # 隐层节点数
        self.output_num = output_num        # 输出节点数
        self.time_step = time_step          # 时间深度
        self.batch_size = batch_size        # batch_size必须能够被time_step整除
        self.learning_rate = learning_rate  # 学习率
        self.lamda = lamda                  # 正则化参数
        self.epochs = epochs                # 迭代次数
        self.model_name = 'LSTM'            # 模型名称
        self.draw_ComputeMap()              # 初始化计算图

    # 绘制计算图
    def draw_ComputeMap(self):
        # LSTM网络输入、输出
        self.input_x = tf.placeholder(tf.float32, [None, self.input_num])
        self.input_y = tf.placeholder(tf.float32, [None, self.output_num])
        # reshape网络输入
        self.input_xx = tf.reshape(self.input_x, [-1, self.time_step, self.input_num])
        # 定义LSTM单元
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)
        # 动态LSTM得到隐层输出
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=self.input_xx,
            initial_state=None,
            dtype=tf.float32,
            time_major=False
        )
        # 接上全连接层
        self.network_out = tf.layers.dense(inputs=self.outputs, units=self.output_num)
        # 预测输出
        self.pred = tf.reshape(self.network_out, [-1, self.output_num])
        # 均方差损失
        self.mse = tf.losses.mean_squared_error(labels=self.input_y, predictions=self.pred)
        # L2正则化项
        self.L2norm = self.lamda * tf.nn.l2_loss(self.lstm_cell.weights[0])
        # 损失函数 = 均方差损失 + L2正则化项
        self.loss = self.mse + self.L2norm
        # 优化器
        self.optm = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # 训练
    def train(self, input_x, input_y):
        dataset = DataSet(input_x, input_y)
        self.sess = tf.Session()  # 会话
        self.sess.run(tf.global_variables_initializer())
        self.batch_num = int(dataset.num_examples / self.batch_size)
        self.cost = []
        # print("\n******** LSTM&DAE开始学习特征 ********")
        for epoch in range(self.epochs):
            total_cost = 0
            for _ in range(self.batch_num):
                train_x, train_y = dataset.next_batch(self.batch_size)  # 读取一个batch的数据
                c, _ = self.sess.run([self.loss, self.optm],
                                     feed_dict={self.input_x: train_x, self.input_y: train_y})
                total_cost += c
            cc = total_cost / self.batch_num
            self.cost.append(cc)
        #     if epoch % 10 == 0:
        #         print('epoch:%5d   train_loss: %.7f' % (epoch, cc))
        # print("******** LSTM&DAE特征提取完成 ********\n")
        # # cost曲线
        # plt.plot(range(len(self.cost)), self.cost)
        # plt.xlabel('Epcoh')
        # plt.ylabel('Loss')
        # plt.title('LSTM模型训练Loss曲线')
        # plt.show()
        # 保存模型
        self.model_path = r'E:\Python Codes\《结构数据解析技术》\SaveModel/LSTM_Regression.ckpt'
        tf.train.Saver().save(self.sess, self.model_path)

    # 预测
    def predict(self, input_x):
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, self.model_path)  # 读取保存的模型
        res = self.sess.run(self.pred, feed_dict={self.input_x: input_x})
        return res

    # 计算指标得分
    def score(self, x, y, y_scaler, print_=True, plot_=True):
        # 得到预测值
        self.y_hat = self.predict(x)
        # 将真实值、预测值反归一化
        y_scale = y_scaler.inverse_transform(y)
        self.y_hat_scale = y_scaler.inverse_transform(self.y_hat)
        # 计算评价指标值
        R_square = r2_score(y_scale, self.y_hat_scale)  # 决定系数
        RMSE = mean_squared_error(y_scale, self.y_hat_scale, squared=False)  # 均方根误差
        if print_:
            print_metrics(self.model_name, R_square, RMSE)  # 输出指标值
        if plot_:
            plot_prediction_and_true(self.model_name, self.y_hat_scale, y_scale)  # 展示拟合效果
        return R_square, RMSE

# LSTM+DAE特征提取
class LSTM_DAE(object):
    def __init__(self, input_num, hidden_num, output_num,
                 time_step, batch_size, learning_rate, lamda,
                 dropout_keep_prob, epochs):
        self.input_num = input_num  # 输入节点数
        self.hidden_num = hidden_num  # 隐层节点数
        self.output_num = output_num  # 输出节点数
        self.time_step = time_step  # 时间深度
        self.batch_size = batch_size  # batch_size必须能够被time_step整除
        self.learning_rate = learning_rate  # 学习率
        self.lamda = lamda  # 正则化参数
        self.dropout_keep_prob = dropout_keep_prob  # dropout参数
        self.epochs = epochs  # 迭代次数
        self.draw_ComputeMap()  # 绘制计算图

    def draw_ComputeMap(self):
        # LSTM网络输入
        self.input_x = tf.placeholder(tf.float32, [None, self.input_num])
        # reshape网络输入
        self.input_xx = tf.reshape(self.input_x, [-1, self.time_step, self.input_num])
        # 定义LSTM单元
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)
        # 动态LSTM得到隐层输出
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=self.input_xx,
            initial_state=None,
            dtype=tf.float32,
            time_major=False
        )
        # 特征
        self.feature = tf.reshape(self.outputs, [-1, self.hidden_num])
        # 加入dropout层
        self.outputs_dropout = tf.nn.dropout(self.outputs, self.dropout_keep_prob)
        # 所有时刻t接上全连接层
        self.network_out = tf.layers.dense(inputs=self.outputs_dropout, units=self.output_num)
        # 重建输入
        self.reconstruction = tf.reshape(self.network_out, [-1, self.output_num])
        # 均方差损失
        self.reconstruction_error = tf.losses.mean_squared_error(
            labels=self.input_x, predictions=self.reconstruction)
        # L2正则化项
        self.L2norm = self.lamda * tf.nn.l2_loss(self.lstm_cell.weights[0])
        # 损失函数 = 均方差损失 + L2正则化项
        self.loss = self.reconstruction_error + self.L2norm
        # 优化器
        self.optm = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_x, input_x_val):
        # 然后开始训练
        dataset = DataSet(input_x, input_x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.batch_num = int(dataset.num_examples / self.batch_size)
            self.cost = []
            # print("\n******** LSTM&DAE开始学习特征 ********")
            for epoch in range(self.epochs):
                total_cost = 0
                for _ in range(self.batch_num):
                    # 读取一个batch的数据
                    train_x, _ = dataset.next_batch(self.batch_size)
                    # 加入噪声，迫使自编码器学习到特征
                    train_x_noise = train_x + 0.3 * np.random.randn(self.batch_size, self.input_num)
                    c, _ = sess.run([self.loss, self.optm], feed_dict={self.input_x: train_x_noise})
                    total_cost += c
                cc = total_cost / self.batch_num
                self.cost.append(cc)
                # if epoch % 50 == 0:
                #     print('epoch:%5d   train_loss: %.7f' % (epoch, cc))
            # print("******** LSTM&DAE特征提取完成 ********\n")
            # # cost曲线
            # plt.plot(range(len(self.cost)), self.cost)
            # plt.xlabel('Epcoh')
            # plt.ylabel('Cost')
            # plt.title('LSTM+DAE: Feature extraction')
            # plt.show()
            # # 评估模型训练效果---计算重建误差
            # print('训练集重建误差：', sess.run(self.reconstruction_error, feed_dict={self.input_x: input_x}))
            # print('验证集重建误差：', sess.run(self.reconstruction_error, feed_dict={self.input_x: input_x_val}))
            # 保存模型
            self.model_path = r'E:\Python Codes\《结构数据解析技术》\SaveModel/LSTM_FeatureExtract.ckpt'
            tf.train.Saver().save(sess, self.model_path)

    def get_feature(self, input_x):
        with tf.Session() as sess1:
            sess1.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess1, self.model_path)  # 读取保存的模型
            Feature = sess1.run(self.feature, feed_dict={self.input_x: input_x})
            return Feature

# 定义DataSet类：包含了next_batch函数
class DataSet(object):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.epochs_completed = 0  # 完成遍历轮数
        self.index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self.num_examples = images.shape[0]  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self.index_in_epochs
        # 初始化：将输入进行洗牌（只在最开始执行一次）
        if self.epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self.num_examples)
            # print(index0)
            np.random.shuffle(index0)
            # print(index0)
            self.images = np.array(self.images)[index0]
            self.labels = np.array(self.labels)[index0]
            # print(self._images)
            # print(self._labels)
            # print("-----------------")

        # *特殊情况：取到最后，剩余样本数不足一个batch_size
        if start + batch_size > self.num_examples:
            # 先把剩余样本的取完
            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            images_rest_part = self.images[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]
            # 重新洗牌，得到新版的数据集
            if shuffle:
                index = np.arange(self.num_examples)
                np.random.shuffle(index)
                self.images = self.images[index]
                self.labels = self.labels[index]
            # 再从新的数据集中取，补全batch_size个样本
            start = 0
            self.index_in_epochs = batch_size - rest_num_examples
            end = self.index_in_epochs
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            # 将旧的和新的拼在一起，得到特殊情况下的batch样本
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)

        # 正常情况：往后取batch_size个样本
        else:
            self.index_in_epochs += batch_size
            end = self.index_in_epochs
            return self.images[start:end], self.labels[start:end]





