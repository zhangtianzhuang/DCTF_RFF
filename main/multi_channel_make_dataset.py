"""
1.获取某个目录下所有文件名以及类标
    Input: dir_name, label
    Output: dict, 类标为键，文件名路径List为value
2.合并数据集
    Input: data1, data2:dict
    Output: data:dict
3.将数据集分为训练集和测试机
    Input: dict:1中的输出, 训练集和测试机的ratio
    Output: dict1, dict2,
4.将训练集和测试分别写入txt文件中
"""
import os
import random
import util
from common_config import *


# 从指定目录获取所有文件名
def get_img_filename(dir_name, label):
    res = []
    for entry in os.scandir(dir_name):
        res.append(dir_name + '/' + entry.name)
    d = dict()
    d[label] = res
    return d


# 合并不同目录的图片
def merge_dataset(data1: dict, data2: dict):
    for k in data2.keys():
        if k in data1.keys():
            data1[k].extend(data2[k])
        else:
            data1[k] = data2[k]
    return data1


# 将数据集分为训练集和测试机
def divide_dataset(data: dict, ratio, data_type):
    if ratio == -1:
        if data_type == 'train':
            return data, dict()
        else:
            return dict(), data
    train_set = dict()
    test_set = dict()
    for k in data.keys():
        imgs = data[k]
        size = len(imgs)
        boundary = int(size * (ratio / (ratio + 1)))
        train_set[k] = imgs[:boundary]
        test_set[k] = imgs[boundary:]
    return train_set, test_set


def write_dataset_to_txt(data, file_path):
    with open(file_path, 'w+') as f:
        for k in data.keys():
            for i in data[k]:
                f.write(i + ' ' + k + '\n')


def get_ratio(is_all: bool):
    if is_all:
        return -1
    else:
        return 4


def get_dirname(dir_name):
    for root, dirs, files in os.walk(dir_name):
        return dirs


if __name__ == '__main__':
    """
    参数说明：
    device_count: 选择设备的数量
    is_all: 是否划分数据集，True表示不划分，必须指定data_type的值，train或者test，表明将所有的数据划分为train_set或者test_set，
            False表示划分数据集，比例参考get_ratio函数中的返回值
    P: 数据的集合，可以添加不同地点的数据
    snr: 选择不同snr的数据
    """
    data_set_type = 'MultiChannelClearedDataset-20'
    root_dir = '/mnt/DiskA-1.7T/ztz/WIFI_Dataset/DCTF_Image/'
    # root_dir = 'D:/WIFI_Dataset/DCTF_Image/'
    devices = common_config_devices
    device_count = len(devices)  # 设备数量
    data_type = 'test'
    is_all = True
    P = ['2']  # '1', '2', '4'
    # P = ['1']
    mylist = []
    first_diff_channel = '1'
    # snr = ['15', '20', '25', 'no']  # '10', '15', '20', '25', '30', '35'
    snr = ['no']
    snr_name = util.build_snr(snr)
    c_name = util.build_device_count(device_count)

    for s in snr:
        dir_name = root_dir + data_set_type + '/snr_' + s + '/Diff_' + first_diff_channel
        for k in P:
            dir_name2 = dir_name + '/P' + k
            for i in range(len(devices)):
                # res = get_img_filename(dir_name2 + '/' + str(i), str(i))
                res = get_img_filename(dir_name2 + '/' + str(devices[i]), str(i+1))
                mylist.append(res)

    data_set = dict()
    for d in mylist:
        data_set = merge_dataset(data_set, d)

    if data_type == 'train':
        for k in data_set.keys():
            random.shuffle(data_set[k])

    train_set, test_set = divide_dataset(data_set, get_ratio(is_all), data_type)
    target = './txt'
    if len(train_set) != 0:
        train_set_file = data_set_type + '_' + util.build_point(P, 'TR') + '_' + c_name + '_' + snr_name
        write_dataset_to_txt(train_set, target + '/' + train_set_file + '.txt')
        print('训练集生成成功：file', train_set_file, 'Size:', len(train_set))
    if len(test_set) != 0:
        test_set_file = data_set_type + '_' + util.build_point(P, 'TE') + '_' + c_name + '_' + snr_name
        write_dataset_to_txt(test_set, target + '/' + test_set_file + '.txt')
        print('测试集生成成功：file', test_set_file, 'Size:', len(test_set))
