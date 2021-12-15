import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import os
from dataset import MyDataset, RawWiFiDataset
from util import *
import util
import matplotlib.pyplot as plt
from common_config import *
import random


# 第三步：训练网络
def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()  # 设置网络为训练模式
    x = list()
    loss_list = list()
    best_epoch = 1
    best_test_acc = 0
    err_count = 0  # 连续错误的次数
    best_acc = 0
    train_err_rate = list()  # 训练时，每epoch的错误率
    test_err_rate = list()  # 测试时，每epoch的错误率
    for epoch in range(1, max_epoch + 1):
        running_loss, train_correct, train_total = 0.0, 0, 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            inputs = inputs.type(torch.FloatTensor)
            inputs, labels = Variable(inputs), Variable(train_labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 将梯度初始化为零
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            # 第二个返回值是找到最大值的索引位置(此处是0,1)
            train_correct += (train_predicted == labels.data).sum()
            train_total += train_labels.size(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:
                # print('[%d] 执行次数 %d' % (epoch, i))
                logger.debug('epoch: {}, batch: {}'.format(epoch, i))
        epoch_loss = running_loss / train_total
        train_acc = 100 * train_correct / train_total
        if train_acc > best_acc:
            best_acc = train_acc
        x.append(epoch)
        loss_list.append(epoch_loss)
        train_err_rate.append(100 - train_acc)
        logger.info('TRAIN: epoch {}, loss: {:.4}, accuracy: {:.4}, best accuracy: {:.4}'.format(
            epoch, epoch_loss, train_acc, best_acc
        ))
        # print('epoch_loss:', epoch_loss)
        # print('train acc:', train_acc)
        # print('train %d epoch loss: %.6f, current accuracy: %.6f   best accuracy:  %.6f' %
        #       (epoch, epoch_loss, train_acc, best_acc))
        # 以下为测试结果
        test_correct, test_total = 0, 0
        test_batch_number = len(test_loader)
        selected_batch_number = test_batch_number
        start = random.randrange(0, test_batch_number - selected_batch_number + 1)
        for i, data in enumerate(test_loader, start):
            if i >= start + selected_batch_number:
                break
            inputs, test_labels = data
            inputs = inputs.type(torch.FloatTensor)
            inputs, labels = Variable(inputs), Variable(test_labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, test_predicted = torch.max(outputs.data, 1)
            test_correct += (test_predicted == labels.data).sum()
            test_total += test_labels.size(0)
        test_acc = 100. * test_correct / test_total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            err_count = 0
            best_epoch = epoch
            logger.info('TEST: accuracy: {:.4}, best accuracy: {:.4}'.format(
                test_acc, best_test_acc
            ))
            # 保存网络模型
            target_dir = '{0}/model/{1}'.format(OUTPUT_DIR, train_data_set_type)
            filename = build_model_para_name_2(net_model_para)
            save_net_model_para(net, target_dir, filename, epoch)
        else:
            err_count += 1
            logger.info('TEST: accuracy: {:.4} less than the best {:.4}, err count: {}'.format(
                test_acc, best_test_acc, err_count
            ))
            # print('current acc less than best acc', 'current test acc:',
            #       test_acc, ', best acc:', best_test_acc, 'err count:', err_count)
        test_err_rate.append(100 - test_acc)
        if err_count >= common_config_early_stopping_max_err_count:
            # print('over-fit, exit now, best epoch is', best_epoch, 'best acc', best_test_acc)
            logger.info('Training END! best train accuracy: {:.4}, best test accuracy: {:.4}, '
                        'best epoch: {}'.format(
                best_acc, best_test_acc, best_epoch
            ))
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


if __name__ == "__main__":
    # 环境
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络模型
    max_epoch = 120
    lr = common_config_learning_rate
    net = get_net(common_config_model).to(device)
    # 模型参数
    batch_size, device_count = common_config_batch_size, common_config_device_count
    train_data_set_type, test_data_set_type = common_config_train_set, common_config_test_set
    train_point, test_point = common_config_train_point, common_config_test_point
    train_snr, test_snr = common_config_train_snr, common_config_test_snr

    # 网络模型文件命名参数
    net_model_para = common_config_net_model_para
    logger = build_logger(os.path.basename(__file__))
    # 训练集
    train_file = build_dataset_file_2(get_train_parameter())
    train_file_list = util.read_list_from_file(train_file)
    train_data = RawWiFiDataset(mat_file=train_file_list, shuffle=True)
    # 测试集
    test_file = build_dataset_file_2(get_test_parameter())
    test_file_list = util.read_list_from_file(test_file)
    test_data = RawWiFiDataset(mat_file=test_file_list)
    # 加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, shuffle=False)
    # 训练
    # 打印参数信息
    logger.info(common_config_info())
    train()
