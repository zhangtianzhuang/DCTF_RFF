import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torchvision
from wifi_net import M1, M2, M3, M4, M5
import torch.nn as nn
from resnet import ResNet18
import torch
from common_config import *
import logging
import logging.handlers

UTIL_OUTPUT_DIR = OUTPUT_DIR


class DatasetParameter(object):
    def __init__(self, type, name, point, device, aug, snr, slice_number, slice_size):
        self.device = device
        self.point = point
        self.slice_size = slice_size
        self.slice_number = slice_number
        self.snr = snr
        self.aug = aug
        self.name = name
        self.type = type


def mkdir_if_not_exist(dir_path):
    if not path.exists(dir_path):
        os.makedirs(dir_path)
        print('folder created successfully, dir = %s' % dir_path)


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues,
                          dir_name='C:/cm', file_name='Default'):
    mkdir_if_not_exist(dir_name)
    saved_path = '{0}/{1}.png'.format(dir_name, file_name)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(saved_path, dpi=600)
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix3(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def build_snr(snrs: list):
    res = 'S'
    for s in snrs:
        res = res + '-' + s
    return res


def build_batch_size(batch_size: int):
    return 'BS-' + str(batch_size)


def build_dataset_file(type: str, point: list, aug: list, snr: list, slice_number, slice_size, name) -> str:
    """
    # $OUTPUT_DIR/class/{数据集}/{TR或者TE的采样地点}_{增扩}_{噪声}_{切片数量}_{切片长度}
    :param type: 数据集类型，must be Train or Test
    :param point: 采样地点
    :param aug: 增扩信道
    :param snr: 加入噪声
    :param slice_number:
    :param slice_size:
    :param name:
    :return:
    """
    dirname = '{}/class/{}'.format(UTIL_OUTPUT_DIR, name)
    filename = '{0}_{1}_{2}_{3}_L{4}-{5}'.format(type, build_point(point), build_aug(aug),
                                                 build_snr(snr), slice_number, slice_size)
    path = '{}/{}.txt'.format(dirname, filename)
    return path


def build_dataset_file_2(dp: DatasetParameter, snr=None):
    if snr is None:
        return build_dataset_file(dp.type, dp.point, dp.aug, dp.snr, dp.slice_number, dp.slice_size, dp.name)
    if type(snr) is not list:
        snr = [snr]
    return build_dataset_file(dp.type, dp.point, dp.aug, snr, dp.slice_number, dp.slice_size, dp.name)


def get_train_parameter() -> DatasetParameter:
    train_para = DatasetParameter(type='Train', name=common_config_train_set, point=common_config_train_point,
                                  device=common_config_devices, aug=common_config_aug,
                                  snr=common_config_train_snr, slice_number=common_config_slice_number,
                                  slice_size=common_config_slice_size)
    return train_para


def get_test_parameter() -> DatasetParameter:
    test_para = DatasetParameter(type='Test', name=common_config_test_set, point=common_config_test_point,
                                 device=common_config_devices, aug=common_config_aug,
                                 snr=common_config_test_snr, slice_number=common_config_slice_number,
                                 slice_size=common_config_slice_size)
    return test_para


def build_point(P: list, type: str = None):
    '''
    :param P:
    :param type: TR or TE
    :return:
    '''
    if type is None or type == '':
        res = 'P'
    else:
        res = type + '-P'

    for p in P:
        res = res + '-' + p
    return res


def build_device_count(c):
    return 'C' + str(c)


def build_epoch(epoch: int):
    return 'EP-' + str(epoch)


def build_aug(aug):
    res = 'A'
    for a in aug:
        res = res + '-' + a
    return res


def build_learning_rate(lr):
    s = str(lr)
    s = s.replace('.', '-')
    return s


def build_model_para_name(model, point, device_count, train_snr, batch_size):
    # 目录：$OUTPUT_DIR/model/{数据集名称}
    # 文件：{网络名称}_{采样地点}_{设备数量}_{噪声}_{batch size}
    res = '{0}_{1}_{2}_{3}_{4}'
    return res.format(model, build_point(point), build_device_count(device_count), build_snr(train_snr),
                      build_batch_size(batch_size))


def build_model_para_name_2(net_model_para: tuple):
    model, point, device_count, train_snr, batch_size = net_model_para
    return build_model_para_name(model, point, device_count, train_snr, batch_size)


def save_net_model_para(net, target_dir, filename, epoch):
    """
    :param net: 网络
    :param target_dir: 存储位置的目录
    :param filename: 文件命名
    :param epoch: 训练轮数
    :return:
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    weight_path = '{}/{}.pth'.format(target_dir, filename)
    torch.save({'epoch': epoch, 'state_dict': net.state_dict()}, weight_path)


def load_net_model_para(target_dir, filename):
    weight_dir = '{}/{}.pth'.format(target_dir, filename)
    net_model = torch.load(weight_dir)['state_dict']
    return net_model


def build_confuse_matrix_name(model_name, test_point, test_snr) -> str:
    # 目录：$OUTPUT_DIR/confuse_matrix/{训练集名称}
    # 网络模型名称，测试集采样地点，噪声
    # 文件：{网络模型名称}_{测试集采样地点}_{噪声}
    return '{0}_{1}_{2}'.format(model_name, test_point, test_snr)


def get_net(name, out_features=9):
    if name == 'M1':
        return M1()
    elif name == 'M2':
        return M2()
    elif name == 'M3':
        return M3()
    elif name == 'M4':
        return M4()
    elif name == 'resnet50':
        net = torchvision.models.resnet50()
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=num_ftrs, out_features=out_features)
        )
        return net
    elif name == 'resnet18':
        net = torchvision.models.resnet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=num_ftrs, out_features=out_features)
        )
        return net
    elif name == 'resnet34':
        net = torchvision.models.resnet50()
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=num_ftrs, out_features=out_features)
        )
        return net
    elif name == 'resnet18-custom':
        return ResNet18()
    elif name == 'M5':
        return M5()


def get_file_name(dir_name: str, suffix: str):
    file_list = os.listdir(dir_name)
    res = list()
    for item in file_list:
        if item.endswith(suffix):
            res.append(os.path.join(dir_name, item))
    return res


def write_dataset_to_txt(data, file_path):
    with open(file_path, 'w+') as f:
        for k in data.keys():
            for i in data[k]:
                f.write(i + ' ' + k + '\n')


def write_list_to_file(data: list, file_path: str):
    with open(file_path, 'w+') as f:
        for line in data:
            f.write(line + '\n')


def read_list_from_file(file_path: str):
    res = list()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.replace('\n', ''))
    return res


def build_logger(filename):
    """
    解放双手：共3种输出日志方式，(1)保存该项目所有日志; (2)按文件打印日志;(3)控制打印日志
    :param filename: 建议传入调用该方法py文件的名字
    :return: logging对象
    """
    log_dir = '{}/logs'.format(UTIL_OUTPUT_DIR)
    log_path = '{}/{}.log'.format(log_dir, filename)
    all_log_path = '{}/all.log'.format(log_dir)
    # 定义日志对象
    logger = logging.getLogger(filename)
    # 定义3种日志输出路径
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_path, mode='a+')
    all_file_handler = logging.FileHandler(filename=all_log_path, mode='a+')
    # 设置不同输出路径打印日志的级别
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)
    all_file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.DEBUG)
    # 设置2种日志输出格式
    file_formatter = logging.Formatter("[%(levelname)s] [%(asctime)s] [%(name)s] >>> %(message)s")
    console_formatter = logging.Formatter(">>> [%(levelname)s] [%(name)s] ### %(message)s")
    # 给3种handler添加输出格式
    file_handler.setFormatter(file_formatter)
    all_file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    # handler绑定到日志对象
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(all_file_handler)
    return logger


if __name__ == '__main__':
    print(build_point(['1', '2']))
