"""
《结构数据解析技术》大作业 --- 论文结果复现："hindcast"方法

日期：2020.4.28

可选择是否打印结果/绘制结果图：
    （1）缺失值、异常值：show1, show2, show3
    （2）模型的训练情况：show_train_process, print_w
    （3）模型的评价结果：print_, plot_
"""
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
# 平均值填充，作为对比
data_stations_fillna1 = data_original.interpolate(method='quadratic')  # 填补缺失值
data_stations1 = station_split(data_stations_fillna1)
data_stations_nan_index1 = data_original.isna()  # 记录缺失值的index
data_stations_nan_index1.iloc[:, 0] = data_original.iloc[:, 0]
data_stations_nan_index1 = station_split(data_stations_nan_index1)

# 展示处理结果
show1 = 0
if show1:
    # 展示填充结果（指定station、指定feature）
    for station_index in np.arange(0, 1):
        for feature_index in np.arange(1, 2):
            show_station_data_fillna(data_stations_fillna, data_stations_nan_index,
                                     station_index, feature_index)
            show_station_data_fillna(data_stations1, data_stations_nan_index1,
                                     station_index, feature_index)
show2 = 0
if show2:
    # 展示填充前后的缺失值分布情况
    show_station_data_nan(data_stations, station_index=0)  # 填充前
    # show_station_data_nan(data_stations_fillna, station_index=0)  # 填充后
# 从缺失值处理后的list型数据集中，提取并组合得到DataFrame型数据集
# ① 按照station顺序排列
data_sort_by_station = pd.DataFrame()
for station_data in data_stations_fillna:
    data_sort_by_station = pd.concat((data_sort_by_station, station_data))
# ② 按照Date顺序排列
data_sort_by_Date = data_sort_by_station.sort_index(axis=0)

# %% 3.异常值处理
# 箱线图可视化：25个station的 “21个feature” 和 “1个目标值”
show3 = 0
if show3:
    box_plot(data_sort_by_Date)

# %% 4.划分训练集、验证集
train_years = [0, 1, 2]   # 训练集数据年份
val_years = [3, 4]  # 验证集数据年份
train_x, train_y, val_x, val_y = train_val_split_by_Date(data_sort_by_Date, train_years, val_years)

# 5.归一化处理
train_x, train_x_scaler = normalize(train_x)  # 训练集特征值
val_x, val_x_scaler = normalize(val_x)        # 验证集特征值
# 对目标值也进行归一化操作
train_y, train_y_scaler = normalize(train_y)  # 训练集目标值
val_y, val_y_scaler = normalize(val_y)        # 验证集目标值

# %% 6.模型评价
models = list(np.zeros((4, 1)))  # 模型
metrics = np.zeros((4, 4))       # 指标
# 6.1 RF模型
models[0] = bias_correction_model(method='RF')  # 模型初始化
models[0].fit(train_x, train_y.reshape(-1, ))   # 训练
metrics[0, 0], metrics[0, 1] = models[0].score(train_x, train_y, train_y_scaler, print_=True, plot_=False)   # 评价模型
metrics[0, 2], metrics[0, 3] = models[0].score(val_x, val_y, val_y_scaler, print_=True, plot_=False)   # 评价模型
# 6.2 SVR模型
models[1] = bias_correction_model(method='SVR')  # 模型初始化
models[1].fit(train_x, train_y.reshape(-1, ))    # 训练
metrics[1, 0], metrics[1, 1] = models[1].score(train_x, train_y, train_y_scaler, print_=True, plot_=False)   # 评价模型
metrics[1, 2], metrics[1, 3] = models[1].score(val_x, val_y, val_y_scaler, print_=True, plot_=False)   # 评价模型
# 6.3 ANN模型
models[2] = bias_correction_model(method='ANN')  # 模型初始化
models[2].fit(train_x, train_y.reshape(-1, ))    # 训练
metrics[2, 0], metrics[2, 1] = models[2].score(train_x, train_y, train_y_scaler, print_=True, plot_=False)   # 评价模型
metrics[2, 2], metrics[2, 3] = models[2].score(val_x, val_y, val_y_scaler, print_=True, plot_=False)   # 评价模型
# 6.4 集成模型
models[3] = bias_correction_ensemble_model(epochs=3000, learning_rate=0.002)  # 模型初始化
models[3].fit(train_x, train_y, show_train_process=False, print_w=False)      # 训练
metrics[3, 0], metrics[3, 1] = models[3].score(train_x, train_y, train_y_scaler, print_=True, plot_=False)   # 评价模型
metrics[3, 2], metrics[3, 3] = models[3].score(val_x, val_y, val_y_scaler, print_=True, plot_=False)   # 评价模型

end = time.time()
print('\n用时：%.2f秒' % (end - start))










