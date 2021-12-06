import itertools
import matplotlib.pyplot as plt
import numpy as np
from common_config import *
import time


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.figure(figsize=(9, 8))
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


if __name__ == '__main__':
    # cnf_matrix = np.array([[2188, 0, 0, 19, 0, 2, 1, 0, 0, 0],
    #                        [0, 2074, 0, 0, 45, 0, 0, 0, 19, 46],
    #                        [0, 0, 2216, 0, 0, 0, 0, 5, 2, 0],
    #                        [9, 0, 0, 2149, 0, 0, 0, 0, 0, 0],
    #                        [0, 34, 0, 0, 1885, 0, 0, 0, 17, 41],
    #                        [5, 0, 0, 0, 0, 1051, 132, 0, 0, 0],
    #                        [2, 0, 0, 0, 0, 39, 2011, 0, 0, 0],
    #                        [0, 4, 0, 0, 0, 0, 0, 1110, 6, 0],
    #                        [0, 72, 0, 0, 215, 0, 0, 1, 2057, 35],
    #                        [0, 8, 0, 0, 67, 0, 0, 0, 35, 2086]])
    # attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R', '1', '2', '3', '4', '5']
    # plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=False,
    #                       title='Normalized confusion matrix')
    plt.figure('Loss')
    x = [1, 2, 3]
    loss_list = [1, 2, 3]
    p = plt.figure('Loss')
    for i in range(1, 5):
        x.append(i)
        loss_list.append(i)
        plt.plot(x, loss_list, 'r-')
        plt.show()
        time.sleep(0.2)
        print(i)

