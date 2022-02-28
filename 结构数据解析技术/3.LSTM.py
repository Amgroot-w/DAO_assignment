"""
《结构数据解析技术》大作业 --- LSTM模型

日期：2020.5.8

"""
# 忽略警告
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import time
from utils import *
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
station_number = 25
metrics = []
# ******* batch_size必须能整除time_step和310 ！！！*******
# time_step和batch_size的组合：(2, 64), (5, 60), (10, 60)
model = lstm(input_num=21, hidden_num=10, output_num=1,
             time_step=10, batch_size=60, learning_rate=0.01,
             lamda=0.00001, epochs=500)
for station_index in range(station_number):
    # 4.划分训练集、验证集
    train_x, train_y, val_x, val_y = train_val_split_by_station(data_stations_fillna, station_index)

    # 5.归一化处理
    train_x, train_x_scaler = normalize(train_x)  # 训练集特征值
    val_x, val_x_scaler = normalize(val_x)        # 验证集特征值
    # 对目标值也进行归一化操作
    train_y, train_y_scaler = normalize(train_y)  # 训练集目标值
    val_y, val_y_scaler = normalize(val_y)        # 验证集目标值

    # 训练
    model.train(train_x, train_y)
    # 模型评价
    R_square1, RMSE1 = model.score(train_x, train_y, train_y_scaler, print_=False, plot_=False)
    R_square2, RMSE2 = model.score(val_x, val_y, val_y_scaler, print_=False, plot_=False)
    metrics.append([R_square1, RMSE1, R_square2, RMSE2])

# %% 展示结果
metrics = np.array(metrics)
# 保存指标值
with open('Results/lstm_metrics.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(metrics)

print('LSTM模型的指标平均值:   训练集: R_square: %.4f, RMSE: %.4f;   验证集: R_square: %.4f, RMSE: %.4f'
      % (metrics[:, 0].mean(), metrics[:, 1].mean(), metrics[:, 2].mean(), metrics[:, 3].mean()))
plot_data = {
    'R_square': metrics[:, 2],
    'RMSE': metrics[:, 3],
    'station_index': np.arange(1, 26)
}
# 设置画图参数
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.4)
# 绘制R_square图
plt.figure(figsize=(25, 5))
sns.barplot(x='station_index', y='R_square', data=plot_data)
plt.xlabel('Station')
plt.ylabel('R_square')
plt.title('R^2 in 25 stations')
plt.savefig(r'Figs/lstm模型25个station的R_square.png')
plt.show()
# 绘制RMSE图
plt.figure(figsize=(25, 5))
sns.barplot(x='station_index', y='RMSE', data=plot_data)
plt.xlabel('Station')
plt.ylabel('RMSE')
plt.title('RMSE in 25 stations')
plt.savefig(r'Figs/lstm模型25个station的RMSE.png')
plt.show()

end = time.time()
print('\n用时：%.2f秒' % (end - start))




