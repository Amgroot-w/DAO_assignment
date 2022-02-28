
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_ss(rmse1, rmse2):
    res = (rmse1 - rmse2) / rmse1
    return res


# %%
losov = pd.read_csv(r'Results/losov_metrics.csv', header=None).values
rf = losov.reshape(25, 4, 4)[:, 0, :]
lstm = pd.read_csv(r'Results/lstm_metrics.csv', header=None).values
lstm_rf = pd.read_csv(r'Results/lstm_rf_metrics.csv', header=None).values
plot_data = {
    'station': list(np.arange(1, 26))*3,
    'R_square': np.row_stack((rf[:, 2], lstm[:, 2], lstm_rf[:, 2])).reshape(-1, ),
    'RMSE': np.row_stack((rf[:, 3], lstm[:, 3], lstm_rf[:, 3])).reshape(-1, ),
    'model_name': ['RF']*25 + ['LSTM']*25 + ['LSTM-RF']*25
}
plt.figure(figsize=(25, 5))
sns.barplot(x='station', y='R_square', hue='model_name', data=plot_data, palette='Set2')
plt.xlabel('station')
plt.ylabel('R_square')
plt.title('R_square in 25 stations')
plt.legend(loc='upper right')
plt.savefig(r'Figs/三种模型在25个station的R_square.png')
plt.show()

plt.figure(figsize=(25, 5))
sns.barplot(x='station', y='RMSE', hue='model_name', data=plot_data, palette='Set2')
plt.xlabel('station')
plt.ylabel('R_MSE')
plt.title('RMSE in 25 stations')
plt.legend(loc='upper right')
plt.savefig(r'Figs/三种模型在25个station的RMSE.png')
plt.show()

























