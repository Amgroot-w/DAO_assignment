
import csv
from utils import *
import time
start = time.time()

# %% 展示结果
def show(metrics):
    print('RF模型的指标平均值:   训练集: R_square: %.4f, RMSE: %.4f;   验证集: R_square: %.4f, RMSE: %.4f'
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
    plt.savefig(r'Figs/lstm-rf 模型25个station的R_square.png')
    plt.show()
    # 绘制RMSE图
    plt.figure(figsize=(25, 5))
    sns.barplot(x='station_index', y='RMSE', data=plot_data)
    plt.xlabel('Station')
    plt.ylabel('RMSE')
    plt.title('RMSE in 25 stations')
    plt.savefig(r'Figs/lstm-rf 模型25个station的RMSE.png')
    plt.show()

    end = time.time()
    print('\n用时：%.2f秒' % (end - start))

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

# 4.划分训练集、验证集
train_years = [0, 1, 2]   # 训练集数据年份
val_years = [3, 4]  # 验证集数据年份
train_x, train_y, val_x, val_y = train_val_split_by_Date(data_sort_by_Date, train_years, val_years)

# 5.归一化处理
train_x, train_x_scaler = normalize(train_x)  # 训练集特征值
val_x, val_x_scaler = normalize(val_x)        # 验证集特征值
# 对目标值也进行归一化操作
train_y, train_y_scaler = normalize(train_y)  # 训练集目标值
val_y, val_y_scaler = normalize(val_y)        # 验证集目标值

paras = np.arange(0, 55, 5)
paras[0] = 1
log = []
metrics = []
for para in paras:
    print('para=', para)
    # 6.模型评价
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=para)  # 模型初始化
    model.fit(train_x, train_y.reshape(-1, ))   # 训练
    train_pred = model.predict(train_x)
    val_pred = model.predict(val_x)

    R_square1, RMSE1 = get_metrics('训练集：RF', train_y, train_pred, train_y_scaler, print_=0, plot_=0)
    R_square2, RMSE2 = get_metrics('验证集：RF', val_y, val_pred, val_y_scaler, print_=0, plot_=0)

    # 记录指标值
    metrics.append([R_square1, RMSE1, R_square2, RMSE2])

    # 记录metrics平均值
    metrics_mean = np.array(metrics).mean(axis=0)
    # show(metrics)  # 展示结果
    log.append(metrics_mean)

log = np.array(log)

with open('Results/RF_hindcast_logs', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(log)

# 画图
plt.plot(range(len(log)), log[:, 0], 'o-', label='Train')
plt.plot(range(len(log)), log[:, 2], 'o-', label='Validation')
plt.xticks(range(len(log)), np.arange(0, 55, 5))
plt.xlabel('min_samples_leaf')
plt.ylabel('R_square')
plt.title('R_square')
plt.legend()
plt.savefig(r'Figs/hindcast：R_square随参数变化的曲线.png')
plt.show()

plt.plot(range(len(log)), log[:, 1], 'o-', label='Train')
plt.plot(range(len(log)), log[:, 3], 'o-', label='Validation')
plt.xticks(range(len(log)), np.arange(0, 55, 5))
plt.xlabel('min_samples_leaf')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.legend()
plt.savefig(r'Figs/hindcast：RMSE随参数变化的曲线.png')
plt.show()

end = time.time()
print('\n用时：%.2f秒' % (end - start))


















