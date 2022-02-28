# %%
from utils import station_split, fillna, mean_squared_error, pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_ss(rmse1, rmse2):
    res = (rmse1 - rmse2) / rmse1
    return res


# %%
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
# 从缺失值处理后的list型数据集中，提取并组合得到DataFrame型数据集
# ① 按照station顺序排列
data_sort_by_station = pd.DataFrame()
for station_data in data_stations_fillna:
    data_sort_by_station = pd.concat((data_sort_by_station, station_data))
# ② 按照Date顺序排列
data_sort_by_Date = data_sort_by_station.sort_index(axis=0)


# %%
ss = []
# *************** LADPS ***************
true = data_sort_by_Date['Next_Tmax']
ladps_pred = data_sort_by_Date['LDAPS_Tmax_lapse']
rmse_ladaps = mean_squared_error(true, ladps_pred, squared=False)

# *************** LR ***************
rmse_lr = 1.5240
ss.append(compute_ss(rmse_ladaps, rmse_lr))

# *************** Ridge ***************
rmse_ridge = 1.5207
ss.append(compute_ss(rmse_ladaps, rmse_ridge))

# *************** SVR ***************
rmse_svr = 1.5343
ss.append(compute_ss(rmse_ladaps, rmse_svr))

# *************** ANN ***************
rmse_ann = 1.6365
ss.append(compute_ss(rmse_ladaps, rmse_ann))

# *************** RF ***************
rmse_rf = 1.1848
ss.append(compute_ss(rmse_ladaps, rmse_rf))

# *************** LSTM ***************
rmse_lstm = 1.5202
ss.append(compute_ss(rmse_ladaps, rmse_lstm))

# *************** LSTM+RF ***************
rmse_lstm_rf = 1.5547
ss.append(compute_ss(rmse_ladaps, rmse_lstm_rf))
model_names = ['LR', 'Ridge', 'SVR', 'ANN', 'RF', 'LSTM', 'LSTM-RF']
# model_names = ['LR', 'Ridge', 'SVR', 'ANN']

plot_data = {
    'ss': ss,
    'name': model_names
}

sns.barplot(x='name', y='ss', data=plot_data)
plt.xlabel('模型')
plt.ylabel('SS指标')
plt.title('各模型的偏差修正效果对比')
plt.show()



























