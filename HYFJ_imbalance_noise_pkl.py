import pandas as pd
import numpy as np
import os
import torch
import pickle
import torch.utils.data as Data

HY_path_list = os.listdir('../data/HYFJ/')
print(HY_path_list)
HY_name = [i.split('.')[0] for i in HY_path_list]
print(HY_name)
HY_PATH = ['../data/HYFJ/' + i for i in HY_path_list]
print(HY_PATH)

# 训练集600个，验证集600个，测试集300个

LABELS = {'BT': 0, 'NF': 1, 'TSW': 2, 'XDBH': 3}
single_len = 1024
SAMPLE_LEN = {'train': 300, 'valid': 300, 'test': 300, 'just_train': 600}
START_INDEX = {'train': 0, 'valid': 600*single_len, 'test': 1200*single_len, 'just_train': 0}
END_INDEX = {'train': (0+SAMPLE_LEN['train'])*single_len, 'valid': (600+SAMPLE_LEN['valid'])*single_len,
             'test': (1200+SAMPLE_LEN['test'])*single_len, 'just_train': (0+SAMPLE_LEN['just_train'])*single_len}

HYFJ_class_num = 4

# 数据不平衡
imbalance_ratio = 0.1

# 信噪比
_snr = 0     # 越小则噪声越大  [-10, -5, 0, 5, 10]


def add_noise(x, d, snr):
    P_signal = np.sum(abs(x)**2)
    P_d = np.sum(abs(d)**2)
    P_noise = P_signal/10**(snr/10)
    noise = np.sqrt(P_noise/P_d)*d
    noise_signal = x.reshape(-1) + noise
    return noise_signal


def HYFJ_data_read(filename, fault_type, data_mode, noise):
    file = np.loadtxt(filename)
    raw_df = pd.DataFrame(file)
    raw_array = np.array(raw_df)

    label = LABELS[fault_type]

    start = START_INDEX[data_mode]
    end = start + single_len
    sample_num = SAMPLE_LEN[data_mode]

    X_list = []
    Y_list = []
    while end <= END_INDEX[data_mode]:
        sample_signal = raw_array[start:end]
        if noise == True:
            temp_noise = np.random.randn(len(sample_signal))
            sample_signal_noise = add_noise(sample_signal, temp_noise, _snr)
            X_list.append(sample_signal_noise)
        else:
            X_list.append(sample_signal)
        Y_list.append(int(label))
        start += single_len
        end += single_len

    X_list = np.float32(X_list)
    # X_tensor = torch.from_numpy(X_list)
    # X_tensor = X_tensor.view(-1, single_len)

    Y_list = np.array(Y_list)
    # Y_tensor = torch.from_numpy(Y_list)

    assert X_list.shape[0] == sample_num

    return X_list, Y_list


def HYFJ_dataset(DATA_MODE, imbalance, NOISE=False):
    the_first_path = HY_PATH[0]
    print(the_first_path)
    the_first_path_name = HY_path_list[0]
    the_first_fault_type = the_first_path_name.split('_')[0]
    X_array, Y_array = HYFJ_data_read(the_first_path, the_first_fault_type, DATA_MODE, noise=NOISE)

    for i in range(1, len(HY_PATH)):
        temp_path = HY_PATH[i]
        print(temp_path)
        temp_path_name = HY_path_list[i]
        temp_fault_type = temp_path_name .split('_')[0]
        temp_X_array, temp_Y_array = HYFJ_data_read(temp_path, temp_fault_type, DATA_MODE, noise=NOISE)

        temp_X_len = temp_X_array.shape[0]
        if imbalance == True and temp_fault_type == 'NF':
            temp_X_array = temp_X_array[:temp_X_len*imbalance_ratio, :]
            temp_Y_array = temp_Y_array[:temp_X_len*imbalance_ratio, :]

        X_array = np.vstack((X_array, temp_X_array))
        Y_array = np.hstack((Y_array, temp_Y_array))


    return X_array, Y_array


def pkl_to_tensorset(filepath):
    file = open(filepath, 'rb')
    temp_list = pickle.load(file)
    X_array = temp_list[0]
    Y_array = temp_list[1]

    X_tensor = torch.from_numpy(X_array)
    X_tensor = X_tensor.view(-1, 1, single_len)

    Y_tensor = torch.from_numpy(Y_array)
    Y_tensor = Y_tensor.long()

    temp_dataset = Data.TensorDataset(X_tensor, Y_tensor)
    return temp_dataset


if __name__ == '__main__':
    mode = 'just_train'
    X_array, Y_array = HYFJ_dataset(mode, imbalance=False, NOISE=True)
    data_pkl = [X_array, Y_array]
    save_path = '../data/HYFJ-' + mode + '_snr-' + str(_snr) + '.pkl'
    f = open(save_path, 'wb')
    pickle.dump(data_pkl, f)
    f.close()
