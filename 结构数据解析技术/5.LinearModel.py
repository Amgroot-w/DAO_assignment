"""
《结构数据解析技术》大作业 --- 线性模型

日期：2020.4.28

只保留LR和Ridge，不要Lasso

"""
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import csv
from utils import *
import time
start = time.time()

# 1.读取数据
data_original = pd.read_csv('Bias_correction_ucl.csv')

# 2.缺失值处理
# 拆分成25个station
data_stations = station_split(data_original)
# 逐个填补缺失值
data_stations_fillna = []
data_stations_nan_index = []
for i in range(25):
    data_stations_fillna.append(fillna(data_stations[i], method='quadratic'))  # 填补缺失值
    data_stations_nan_index.append(data_stations[i].isna())  # 记录缺失值的index
# 展示处理结果
show1 = False
if show1:
    # 展示填充结果（指定station、指定feature）
    for station_index in np.arange(0, 2):
        for feature_index in np.arange(4, 8):
            show_station_data_fillna(data_stations_fillna, data_stations_nan_index,
                                     station_index, feature_index)
show2 = False
if show2:
    # 展示填充前后的缺失值分布情况
    show_station_data_nan(data_stations, station_index=1)
    show_station_data_nan(data_stations_fillna, station_index=1)
# 从缺失值处理后的list型数据集中，提取并组合得到DataFrame型数据集
# ① 按照station顺序排列
data_sort_by_station = pd.DataFrame()
for station_data in data_stations_fillna:
    data_sort_by_station = pd.concat((data_sort_by_station, station_data))
# ② 按照Date顺序排列
data_sort_by_Date = data_sort_by_station.sort_index(axis=0)

# 3.异常值处理
# 箱线图可视化：25个station的 “21个feature” 和 “1个目标值”
show3 = False
if show3:
    box_plot(data_sort_by_Date)

# 开始按station迭代
losov_info = {'models': [], 'metrics': [], 'model_names': [], 'station_index': []}
station_number = 25
for station_index in range(station_number):

    # 4.划分训练集、验证集
    train_x, train_y, val_x, val_y = train_val_split_by_station(data_stations_fillna, station_index)

    # 5.归一化处理
    train_x, train_x_scaler = normalize(train_x)  # 训练集特征值
    val_x, val_x_scaler = normalize(val_x)        # 验证集特征值
    # 对目标值也进行归一化操作
    train_y, train_y_scaler = normalize(train_y)  # 训练集目标值
    val_y, val_y_scaler = normalize(val_y)        # 验证集目标值

    # 6.模型训练
    models = list(np.zeros((4, 1)))  # 模型
    metrics = np.zeros((4, 4))       # 指标

    # 6.1 训练LR模型
    models[0] = LinearRegression()  # 模型初始化
    models[0].fit(train_x, train_y.reshape(-1, ))   # 训练
    train_pred = models[0].predict(train_x)  # 预测
    val_pred = models[0].predict(val_x)  # 预测
    metrics[0, 0], metrics[0, 1] = get_metrics('LR', train_y, train_pred, train_y_scaler)
    metrics[0, 2], metrics[0, 3] = get_metrics('LR', val_y, val_pred, val_y_scaler)

    # 6.2 训练Ridge模型
    models[1] = Ridge()  # 模型初始化
    models[1].fit(train_x, train_y.reshape(-1, ))   # 训练
    train_pred = models[1].predict(train_x)  # 预测
    val_pred = models[1].predict(val_x)  # 预测
    metrics[1, 0], metrics[1, 1] = get_metrics('Ridge', train_y, train_pred, train_y_scaler)
    metrics[1, 2], metrics[1, 3] = get_metrics('Ridge', val_y, val_pred, val_y_scaler)

    # 记录训练数据
    losov_info['models'].append(models)
    losov_info['metrics'].append(metrics)
    losov_info['model_names'].append(['LR', 'Ridge', '///', '///'])
    losov_info['station_index'].append([station_index+1] * 4)
    # 打印训练结果
    print('Training ensemble model：%d/25 ' % (station_index+1))
print('训练完成！\n')

# 7.模型评价
losov_print1(losov_info)  # 打印结果
losov_plot(losov_info)   # 画图展示

# 保存指标值
metrics = np.array(losov_info['metrics']).reshape(-1, 4)
with open('Results/LinearModel_metrics.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(metrics)

end = time.time()
print('\n用时：%.2f秒' % (end - start))















