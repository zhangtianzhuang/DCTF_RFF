import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torchvision
from wifi_net import M1, M2, M3, M4, M5
import torch.nn as nn
from resnet import ResNet18


def mkdir_if_not_exist(dir_path):
    if not path.exists(dir_path):
        os.makedirs(dir_path)
        print('folder created successfully, dir = %s' % dir_path)


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues,
                          dir_name='C:/cm', file_name='Default'):
    mkdir_if_not_exist(dir_name)
    saved_path = dir_name + '/' + file_name + '.png'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
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


def buildSNR(snrs: list):
    res = 'S'
    for s in snrs:
        res = res + '-' + s
    return res


def buildP(P: list, type: str):
    '''
    :param P:
    :param type: TR or TE
    :return:
    '''
    res = type + '-P'
    for p in P:
        res = res + '-' + p
    return res


def buildC(c):
    return 'C' + str(c)


def build_aug(aug):
    res = 'A'
    for a in aug:
        res = res + '-' + a
    return res


def buildLR(lr):
    s = str(lr)
    s = s.replace('.', '-')
    return s


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


if __name__ == '__main__':
    # test buildLR
    # s = buildLR(0.0001)
    # print(s)

    # res = get_file_name('/home/bjtu/ztz/project/DCTF_RFF/main', '.py')
    # print(res)
    # print(len(res))
    res = read_list_from_file('/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/ClearedDataset-1-RawSlice/TE-P-1-2_A-no_S-5-10-15-20-25-30-no_L10-128.txt')
    print(res)
