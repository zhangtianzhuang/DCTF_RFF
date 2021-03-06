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
from common_config import *
import random

model = common_config_model
max_epoch = 120
lr = common_config_learning_rate
net = util.get_net(model)
net = net.cuda()

epochs_name = str(max_epoch)

# batch_size = common_config_batch_size
# device_count = common_config_device_count
# train_data_set_type = common_config_train_set
# test_data_set_type = common_config_test_set
# train_point = common_config_train_point
# test_point = common_config_test_point
# train_snr = common_config_train_snr
# test_snr = common_config_test_snr
batch_size, device_count = common_config_batch_size, common_config_device_count
train_data_set_type, test_data_set_type = common_config_train_set, common_config_test_set
train_point, test_point = common_config_train_point, common_config_test_point
train_snr, test_snr = common_config_train_snr, common_config_test_snr

snr_name_train = util.build_snr(train_snr)
snr_name_test = util.build_snr(test_snr)

c_name = util.build_device_count(device_count)


def save_weights(epoch, filename):
    """
    Args:
        :param epoch:
        :param filename:
    """
    weight_dir = OUTPUT_DIR + '/model/' + train_data_set_type
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict()},
               '%s/%s' % (weight_dir, filename))


# 第三步：训练网络
def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    x = list()
    loss_list = list()
    best_epoch = 1
    best_test_acc = 0
    err_count = 0  # 连续错误的次数
    best_acc = 0
    train_err_rate = list()  # 训练时，每epoch的错误率
    test_err_rate = list()  # 测试时，每epoch的错误率
    for epoch in range(1, max_epoch + 1):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            inputs, labels = Variable(inputs), Variable(train_labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()  # 将梯度初始化为零
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.detach(), 1)
            # 第二个返回值是找到最大值的索引位置(此处是0,1)
            train_correct += (train_predicted == labels.data).sum()
            train_total += train_labels.size(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d] 执行次数 %d' % (epoch, i))
        epoch_loss = running_loss / train_total
        print('epoch_loss:', epoch_loss)
        train_acc = 100 * train_correct / train_total
        print('train acc:', train_acc)
        if train_acc > best_acc:
            best_acc = train_acc
        x.append(epoch)
        loss_list.append(epoch_loss)
        train_err_rate.append(100 - train_acc)
        print('train %d epoch loss: %.6f, current accuracy: %.6f   best accuracy:  %.6f' %
              (epoch, epoch_loss, train_acc, best_acc))
        # 以下为测试结果
        test_correct = 0
        test_total = 0
        test_batch_number = len(test_loader)
        selected_batch_number = test_batch_number
        start = random.randrange(0, test_batch_number-selected_batch_number+1)
        for i, data in enumerate(test_loader, start):
            if i >= start + selected_batch_number:
                break
            inputs, test_labels = data
            inputs, labels = Variable(inputs), Variable(test_labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, test_predicted = torch.max(outputs.data, 1)
            test_correct += (test_predicted == labels.data).sum()
            test_total += test_labels.size(0)
            if i % 50 == 0:
                print('test i', i)
        test_acc = test_correct / test_total * 100
        if test_acc > best_test_acc:
            print('current test acc:', test_acc, ', best acc:', best_test_acc)
            best_test_acc = test_acc
            err_count = 0
            best_epoch = epoch
            save_weights(epoch, build_para_name())
        else:
            err_count += 1
            print('current acc less than best acc', 'current test acc:',
                  test_acc, ', best acc:', best_test_acc, 'err count:', err_count)
        test_err_rate.append(100 - test_acc)
        if err_count >= common_config_early_stopping_max_err_count:
            print('over-fit, exit now, best epoch is', best_epoch, 'best acc', best_test_acc)
            plt.figure('Error rate')
            plt.plot(x, train_err_rate, 'b--o', label='Train Error Rate(%)')
            plt.plot(x, test_err_rate, 'g:^', label='Test Error Rate(%)')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('error rate(%)')
            plt.show()

            plt.figure('Loss')
            plt.plot(x, loss_list, 'r-', label='Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.show()
            return


def build_para_name():
    return model + '_' + train_common + '_BS_' + str(common_config_batch_size) + '_LR' + util.build_learning_rate(lr) + '.pth'


if __name__ == "__main__":
    # Config
    train_common = train_data_set_type + '_' + util.build_point(train_point, 'TR') \
                   + '_' + c_name + '_' + snr_name_train
    # 训练后的模型参数命名
    para = build_para_name()
    train_txt_dir = OUTPUT_DIR + '/class' + '/' + train_data_set_type + '/'
    test_txt_dir = OUTPUT_DIR + '/class' + '/' + test_data_set_type + '/'
    train_set_file = train_txt_dir + train_common + '.txt'
    test_set_file = test_txt_dir + test_data_set_type + '_' \
                    + util.build_point(test_point, 'TE') + '_' + c_name + '_' \
                    + snr_name_test + '.txt'
    cm_name = model + '_' + c_name + '_' + util.build_point(train_point, 'TR') + '_TR' + util.build_snr(train_snr) \
              + '_' + util.build_point(test_point, 'TE') + '_TE' + util.build_snr(test_snr) \
              + '_EP' + epochs_name \
              + '_LR' + util.build_learning_rate(lr)
    print('Train filename', train_set_file)
    print('Test filename', test_set_file)
    train_data = MyDataset(txt=train_set_file, type='train', device_count=device_count,
                           transform=transforms.Compose([transforms.CenterCrop(64),
                                                         transforms.ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_data = MyDataset(txt=test_set_file, type='test', device_count=device_count, shuffle=True,
                          transform=transforms.Compose([transforms.CenterCrop(64),
                                                        transforms.ToTensor()]))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2,
                             shuffle=False, pin_memory=True)
    train()
