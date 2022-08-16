import scipy.io as sio
import numpy as np


# 加载.mat数据dB化并标准化
def load_data(signal_file, label_file):
    signal_data = sio.loadmat(signal_file)
    label_data = sio.loadmat(label_file)

    signal = signal_data[signal_file]
    label = label_data[label_file]

    signal = np.expand_dims(signal, axis=2)  # 用于扩展数组的形状，在2位置添加数据
    signal = signal.astype(np.float32)
    label = label.astype(np.int)
    signal_shape = signal.shape  # 原始数据保存
    # RCS数据dB化并标准化
    for j in range(signal_shape[0]):
        signal[j, :, :] = 10 * np.log10(signal[j, :, :])
        mean = signal[j, :, :].mean(axis=0)
        signal[j, :, :] -= mean
        std = signal[j, :, :].std(axis=0)
        signal[j, :, :] /= std
    return signal, label


# 类别标签转换为one_hot向量
def one_hot_transform(label, dimension):
    ont_hot_array = np.zeros([1, dimension], dtype=np.int)
    ont_hot_array[0, label - 1] = 1
    return ont_hot_array


# 读取数据和标签并划分样本
def read_data(signal_file, label_file, ntrain_per_class):
    signal, label = load_data(signal_file, label_file)

    n_class = np.max(label)
    shape = np.shape(signal)

    train_num = ntrain_per_class * n_class
    test_num = shape[0] - train_num
    train_x = np.zeros([train_num, shape[1], shape[2]], dtype=np.float32)
    train_y = np.zeros([train_num, n_class], dtype=np.int)
    test_x = np.zeros([test_num, shape[1], shape[2]], dtype=np.float32)
    test_y = np.zeros([test_num, n_class], dtype=np.int)

    test_index = 0

    for i in range(n_class):

        temprow, tempcol = np.where(label == i + 1)
        shuffle_number = np.arange(len(temprow))
        np.random.shuffle(shuffle_number)

        for j in range(ntrain_per_class):
            train_x[i * ntrain_per_class + j, :, :] = \
                signal[temprow[shuffle_number[j]], :, :]
            train_y[i * ntrain_per_class + j, :] = \
                one_hot_transform(label[temprow[shuffle_number[j]], 0], n_class)
        for j in range(ntrain_per_class, len(temprow)):
            test_x[test_index + j - ntrain_per_class, :, :] = \
                signal[temprow[shuffle_number[j]], :, :]
            test_y[test_index + j - ntrain_per_class, :] = \
                one_hot_transform(label[temprow[shuffle_number[j]], 0], n_class)

        test_index = test_index + len(temprow) - ntrain_per_class

    return (train_x, train_y), (test_x, test_y)
