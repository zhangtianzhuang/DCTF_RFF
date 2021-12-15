import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import os
from dataset import MyDataset
import util
import matplotlib.pyplot as plt
import torchvision
from common_config import *
from multi_channel_dataset import MultiChannelDataset
import logging
import random

switch_flag = [False, False, False, False]

Model = common_config_model
lr = common_config_learning_rate
net = util.get_net(Model)
net = net.cuda()
batch_size = common_config_batch_size
device_count = common_config_device_count

train_data_set_type = common_config_train_set
test_data_set_type = common_config_test_set
P = common_config_train_point
testP = common_config_test_point
snr = common_config_train_snr
test_snr = common_config_test_snr
snr_name_train = util.build_snr(snr)
snr_name_test = util.build_snr(test_snr)
c_name = util.build_device_count(device_count)


def save_weights(epoch, filename):
    """
    Args:
        :param epoch:
        :param filename:
    """
    # weight_dir = os.path.join('./output/transformModel', 'train', 'weights')
    # weight_dir = os.path.join('./output')
    weight_dir = OUTPUT_DIR + '/model/' + train_data_set_type
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict()},
               '%s/%s' % (weight_dir, filename))


# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                                weight_decay=5e-4)
optimizers = list()
learning_rates = [0.0001]
switch_point = [20, 40]
max_epoch = switch_point[-1]
epochs_name = str(max_epoch)

for lr in learning_rates:
    optimizers.append(optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4))


def train():
    global optimizer
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 70.0
    x, train_x, train_y, test_x, test_y = list(), list(), list(), list(), list()
    iter_num = 0  # 训练batch的次数
    for epoch in range(1, max_epoch + 1):
        net.train()
        if epoch >= switch_point[0] and not switch_flag[0]:
            optimizer = optimizers[0]
            switch_flag[0] = True
        # if epoch >= switch_point[1] and not switch_flag[1]:
        #     print('>>切换到LR=0.001')
        #     optimizer = optimizers[1]
        #     switch_flag[1] = True
        # if epoch >= switch_point[2] and not switch_flag[2]:
        #     print('>>切换到LR=0.0001')
        #     optimizer = optimizers[2]
        #     switch_flag[2] = True
        sum_loss = 0.0
        train_correct_num = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            iter_num += 1
            inputs, train_labels = data
            inputs, labels = Variable(inputs), Variable(train_labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()  # 将梯度初始化为零
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)

            train_correct_num += (train_predicted == labels.data).sum()
            train_total += train_labels.size(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            train_acc = 100. * train_correct_num / train_total
            print('[epoch: %d, iter: %d] Loss: %.03f | Acc: %.3f' % (epoch, iter_num, sum_loss / (i + 1), train_acc))

        # epoch_loss = sum_loss / train_total
        # print('epoch_loss:', epoch_loss)
        # train_acc = 100 * train_correct_num / train_total
        # print('train acc:', train_acc)
        # if train_acc > best_acc:
        #     best_acc = train_acc
        # x.append(epoch)
        # loss_list.append(epoch_loss)
        # print('train %d epoch loss: %.6f, current accuracy: %.6f   best accuracy:  %.6f' %
        #       (epoch, epoch_loss, train_acc, best_acc))
        # 以下为测试结果
        print("Waiting Test!")
        with torch.no_grad():
            test_correct_num = 0
            test_total = 0
            net.eval()
            for i, data in enumerate(test_loader, 0):
                inputs, test_labels = data
                inputs, labels = Variable(inputs), Variable(test_labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                _, test_predicted = torch.max(outputs.data, 1)
                test_correct_num += (test_predicted == labels.data).sum()
                test_total += test_labels.size(0)
            test_acc = 100. * test_correct_num / test_total
            print('Test Accuracy: %.03f' % test_acc)
            if test_acc > best_test_acc:
                print('>>> Test Accuracy Update!')
                best_test_acc = test_acc
                save_weights(epoch, build_para_name())
    # plt.figure('Error rate')
    # plt.plot(x, train_err_rate, 'b--o', label='Train Error Rate(%)')
    # plt.plot(x, test_err_rate, 'g:^', label='Test Error Rate(%)')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('error rate(%)')
    # plt.show()
    # plt.figure('Loss')
    # plt.plot(x, loss_list, 'r-', label='Loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()
    # return


def build_para_name():
    return Model + '_' + train_common + '_BS_' + str(common_config_batch_size) + '_LR' + util.build_learning_rate(lr) + '.pth'


if __name__ == "__main__":
    # Config
    train_common = train_data_set_type + '_' + util.build_point(P, 'TR') \
                   + '_' + c_name + '_' + snr_name_train
    # 训练后的模型参数命名
    para = build_para_name()
    train_txt_dir = OUTPUT_DIR + '/class' + '/' + train_data_set_type + '/'
    test_txt_dir = OUTPUT_DIR + '/class' + '/' + test_data_set_type + '/'
    train_set_file = train_txt_dir + train_common + '.txt'
    test_set_file = test_txt_dir + test_data_set_type + '_' + \
                    util.build_point(testP, 'TE') + '_' + c_name + '_' + \
                    snr_name_test + '.txt'
    cm_name = Model + '_' + c_name + '_' + util.build_point(P, 'TR') + '_TR' + util.build_snr(snr) + \
              '_' + util.build_point(testP, 'TE') + '_TE' + util.build_snr(test_snr) + \
              '_EP' + epochs_name + \
              '_LR' + util.build_learning_rate(lr)
    print('Model name is', para)
    print('Train filename', train_set_file)
    print('Test filename', test_set_file)
    print('CM Name', cm_name)
    # 第一步：数据加载与处理,选择训练设备cpu/gpu
    train_data = MyDataset(txt=train_set_file, type='train', device_count=device_count,
                           transform=transforms.Compose([transforms.CenterCrop(64),
                                                         transforms.ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_data = MyDataset(txt=test_set_file, type='test', device_count=device_count, shuffle=True,
                          transform=transforms.Compose([transforms.CenterCrop(64),
                                                        transforms.ToTensor()]))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2,
                             shuffle=False, pin_memory=True)
    print('create data_loader successfully!')
    print('parameters:', '\ndata_type:', train_common, '\nbatch_size:', batch_size)
    train()
