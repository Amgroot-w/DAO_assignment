"""
《深度学习》作业 --- LSTM的应用：预测飞机月流量

2020.11.20

n vs.1 型LSTM

"""
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cap
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 随机数种子
np.random.seed(6)

# %% 导入数据
# 12年，每年12个月，共144个月的流量数据
true = pd.read_csv('plane_data.csv').values[:, 1]
data = pd.read_csv('plane_data.csv').values

# 归一化处理
maximum, minimum = max(data[:, 1]), min(data[:, 1])
data[:, 1] = (data[:, 1] - minimum) / (maximum - minimum)

# %% 建立LSTM数据集 -- 飞机月流量数据
T = 10  # LSTM的时间深度T
data_x = []
data_y = []
for i in range(len(data) - T):
    X = data[i:i + T, 1]
    data_x.append(X)
    data_y.append(data[i + T, 1])
data_x = np.array(data_x)
data_y = np.array(data_y)
m = data_x.shape[0]  # LSTM的训练样本数

# %% 配置网络参数
input_dim = 1   # 输入维度
hidden_dim = 8  # 隐层输出维度
cell_dim = 8    # 细胞状态维度
output_dim = 1  # 输出维度
alpha = 0.5       # 学习率
epochs = 100    # 迭代次数

# %% 参数初始化
sigma = 0.5
w_fh = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_ih = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_ch = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_oh = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])

w_fx = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_ix = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_cx = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_ox = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])

b_f = np.zeros([1, cell_dim])
b_i = np.zeros([1, cell_dim])
b_c = np.zeros([1, cell_dim])
b_o = np.zeros([1, cell_dim])

v = np.random.uniform(-sigma, sigma, [hidden_dim, output_dim])
c = np.zeros([1, output_dim])

# %% 训练
cost = []
for epoch in range(epochs):
    for k in range(m):
        # 读取第i次训练的输出y
        input_y = np.atleast_2d(data_y[k])

        # 初始化
        # 储存每个时刻的加权输入
        net = {
            'f': np.zeros([T, cell_dim]),
            'i': np.zeros([T, cell_dim]),
            'c': np.zeros([T, cell_dim]),
            'o': np.zeros([T, cell_dim])
        }
        # 储存每个时刻的门、候选细胞状态
        door = {
            'f': np.zeros([T, cell_dim]),
            'i': np.zeros([T, cell_dim]),
            'c': np.zeros([T, cell_dim]),
            'o': np.zeros([T, cell_dim])
        }
        # 储存每个时刻的隐层状态、细胞状态
        state = {
            'hidden': np.zeros([T, hidden_dim]),
            'cell': np.zeros([T, cell_dim])
        }

        # 初始化各参数梯度
        d_wfh = np.zeros_like(w_fh)
        d_wfx = np.zeros_like(w_fx)
        d_wih = np.zeros_like(w_ih)
        d_wix = np.zeros_like(w_ix)
        d_wch = np.zeros_like(w_ch)
        d_wcx = np.zeros_like(w_cx)
        d_woh = np.zeros_like(w_oh)
        d_wox = np.zeros_like(w_ox)
        d_bf = np.zeros_like(b_f)
        d_bi = np.zeros_like(b_i)
        d_bc = np.zeros_like(b_c)
        d_bo = np.zeros_like(b_o)

        # forward propagation
        for t in range(T):
            # 得到t时刻的输入x、t-1时刻的隐层状态、t-1时刻的细胞状态
            input_x = np.atleast_2d(data_x[k, t])
            if t == 0:
                cell_pre = np.zeros([1, cell_dim])
                hidden_pre = np.zeros([1, hidden_dim])
            else:
                cell_pre = state['cell'][t-1:t, :]
                hidden_pre = state['hidden'][t-1:t, :]

            # t时刻的：加权输入
            net_f = np.matmul(hidden_pre, w_fh.T) + np.matmul(input_x, w_fx.T) + b_f
            net_i = np.matmul(hidden_pre, w_ih.T) + np.matmul(input_x, w_ix.T) + b_i
            net_c = np.matmul(hidden_pre, w_ch.T) + np.matmul(input_x, w_cx.T) + b_c
            net_o = np.matmul(hidden_pre, w_oh.T) + np.matmul(input_x, w_ox.T) + b_o

            # t时刻的：门、候选细胞状态
            f = cap.sigmoid(net_f)
            i = cap.sigmoid(net_i)
            cell = cap.tanh(net_c)
            o = cap.sigmoid(net_o)

            # t时刻的：隐层状态、细胞状态
            cell_now = f * cell_pre + i * cell
            hidden_now = o * cap.tanh(cell_now)

            # 保存中间变量
            net['f'][t, :] = net_f
            net['i'][t, :] = net_i
            net['c'][t, :] = net_c
            net['o'][t, :] = net_o

            door['f'][t, :] = f
            door['i'][t, :] = i
            door['c'][t, :] = cell
            door['o'][t, :] = o

            state['cell'][t, :] = cell_now
            state['hidden'][t, :] = hidden_now

            # 最后一层计算输出
            if t == T - 1:
                network_out = np.matmul(hidden_now, v) + c
                pred = network_out  # 线性输出

                error = 1/2 * (pred - input_y) ** 2
                output_delta = pred - input_y

        # 记录误差
        cost.append(error[0, 0])

        # back prapagation through time (BPTT)
        for t in reversed(range(T)):
            # 得到时刻t的输入x
            input_x = np.atleast_2d(data_x[k, t])

            # 计算隐层误差 spcae
            # 只有最后一层有空间上反向传播的误差
            if t == T - 1:
                hidden_delta_space = np.matmul(output_delta, v.T)
            else:
                hidden_delta_space = np.zeros([1, cell_dim])

            # 计算隐层误差 time
            # 只有最后一层没有有时间上反向传播的误差
            if t == T - 1:
                hidden_delta_time = np.zeros([1, cell_dim])
            else:
                # 根据上一层的4个小误差，算本层的BPTT误差
                hidden_delta_time = np.matmul(delta_f, w_fh) + np.matmul(delta_i, w_ih) + \
                                    np.matmul(delta_c, w_ch) + np.matmul(delta_o, w_oh)

            # 计算总误差（时间+空间）
            hidden_delta = hidden_delta_space + hidden_delta_time

            # 计算4个小误差
            if t == 0:
                delta_f = np.zeros([1, cell_dim])
            else:
                delta_f = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                          * state['cell'][t - 1, :] * door['f'][t, :] * (1 - door['f'][t, :])

            delta_i = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                      * door['c'][t, :] * door['f'][t, :] * (1 - door['f'][t, :])

            delta_c = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                      * door['i'][t, :] * (1 - door['c'][t, :] ** 2)

            delta_o = hidden_delta * cap.tanh(door['c'][t, :]) \
                      * door['o'][t, :] * (1 - door['o'][t, :])

            # 计算梯度
            if t == 0:
                d_wfh += np.matmul(delta_f.T, np.zeros([1, cell_dim]))
                d_wih += np.matmul(delta_i.T, np.zeros([1, cell_dim]))
                d_wch += np.matmul(delta_c.T, np.zeros([1, cell_dim]))
                d_woh += np.matmul(delta_o.T, np.zeros([1, cell_dim]))
            else:
                d_wfh += np.matmul(delta_f.T, state['hidden'][t-1:t, :])
                d_wih += np.matmul(delta_i.T, state['hidden'][t-1:t, :])
                d_wch += np.matmul(delta_c.T, state['hidden'][t-1:t, :])
                d_woh += np.matmul(delta_o.T, state['hidden'][t-1:t, :])

            d_wfx += np.matmul(delta_f.T, input_x)
            d_wix += np.matmul(delta_i.T, input_x)
            d_wcx += np.matmul(delta_c.T, input_x)
            d_wox += np.matmul(delta_o.T, input_x)

            d_bf += delta_f
            d_bi += delta_i
            d_bc += delta_c
            d_bo += delta_o

        # 参数更新
        w_fh -= alpha * d_wfh
        w_ih -= alpha * d_wih
        w_ch -= alpha * d_wch
        w_oh -= alpha * d_woh

        w_fx -= alpha * d_wfx
        w_ix -= alpha * d_wix
        w_cx -= alpha * d_wcx
        w_ox -= alpha * d_wox

        b_f -= alpha * d_bf
        b_i -= alpha * d_bi
        b_c -= alpha * d_bc
        b_o -= alpha * d_bo

    if epoch % 10 == 0:
        print('Epoch:%3d  Cost:%.6f' % (epoch, error))

plt.plot(range(len(cost)), cost)
plt.xlabel('迭代次数')
plt.ylabel('Loss')
plt.title('误差变化曲线')
plt.show()

#%% 测试
pred = list(data[:T, 1])

for k in range(m):
    # 读取第i次训练的输出y
    input_y = np.atleast_2d(data_y[k])
    # 初始化
    # 储存每个时刻的加权输入
    net = {
        'f': np.zeros([T, cell_dim]),
        'i': np.zeros([T, cell_dim]),
        'c': np.zeros([T, cell_dim]),
        'o': np.zeros([T, cell_dim])
    }
    # 储存每个时刻的门、候选细胞状态
    door = {
        'f': np.zeros([T, cell_dim]),
        'i': np.zeros([T, cell_dim]),
        'c': np.zeros([T, cell_dim]),
        'o': np.zeros([T, cell_dim])
    }
    # 储存每个时刻的隐层状态、细胞状态
    state = {
        'hidden': np.zeros([T, hidden_dim]),
        'cell': np.zeros([T, cell_dim])
    }
    # 初始化各参数梯度
    d_wfh = np.zeros_like(w_fh)
    d_wfx = np.zeros_like(w_fx)
    d_wih = np.zeros_like(w_ih)
    d_wix = np.zeros_like(w_ix)
    d_wch = np.zeros_like(w_ch)
    d_wcx = np.zeros_like(w_cx)
    d_woh = np.zeros_like(w_oh)
    d_wox = np.zeros_like(w_ox)
    d_bf = np.zeros_like(b_f)
    d_bi = np.zeros_like(b_i)
    d_bc = np.zeros_like(b_c)
    d_bo = np.zeros_like(b_o)

    # forward propagation
    for t in range(T):
        # 得到t时刻的输入x、t-1时刻的隐层状态、t-1时刻的细胞状态
        input_x = np.atleast_2d(data_x[k, t])
        if t == 0:
            cell_pre = np.zeros([1, cell_dim])
            hidden_pre = np.zeros([1, hidden_dim])
        else:
            cell_pre = state['cell'][t-1:t, :]
            hidden_pre = state['hidden'][t-1:t, :]

        # t时刻的：加权输入
        net_f = np.matmul(hidden_pre, w_fh.T) + np.matmul(input_x, w_fx.T) + b_f
        net_i = np.matmul(hidden_pre, w_ih.T) + np.matmul(input_x, w_ix.T) + b_i
        net_c = np.matmul(hidden_pre, w_ch.T) + np.matmul(input_x, w_cx.T) + b_c
        net_o = np.matmul(hidden_pre, w_oh.T) + np.matmul(input_x, w_ox.T) + b_o

        # t时刻的：门、候选细胞状态
        f = cap.sigmoid(net_f)
        i = cap.sigmoid(net_i)
        cell = cap.tanh(net_c)
        o = cap.sigmoid(net_o)

        # t时刻的：隐层状态、细胞状态
        cell_now = f * cell_pre + i * cell
        hidden_now = o * cap.tanh(cell_now)

        # 保存中间变量
        net['f'][t, :] = net_f
        net['i'][t, :] = net_i
        net['c'][t, :] = net_c
        net['o'][t, :] = net_o

        door['f'][t, :] = f
        door['i'][t, :] = i
        door['c'][t, :] = cell
        door['o'][t, :] = o

        state['cell'][t, :] = cell_now
        state['hidden'][t, :] = hidden_now

        # 最后一层计算输出
        if t == T - 1:
            network_out = np.matmul(hidden_now, v) + c
            pred.append(network_out[0, 0])

pred = np.array(pred).reshape(true.shape)
pred = minimum + (maximum - minimum) * pred  # 反归一化

plt.plot(range(pred.shape[0]), pred, label='预测值')
plt.plot(range(pred.shape[0]), true, label='真实值')
plt.xlabel('月数')
plt.ylabel('月流量')
plt.title('飞机月流量预测')
plt.legend()
plt.show()

print('预测误差：%.4f' % np.mean(np.abs(pred - true)))
































