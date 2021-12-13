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
from torchvision.datasets import cifar
from torchvision.datasets import imagenet
import torchvision
from common_config import *
from multi_channel_dataset import MultiChannelDataset

imagenet_dateset = imagenet.ImageNet(OUTPUT_DIR + '/dataset/picdata', download=True, transform=transforms)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# trainData = cifar.CIFAR10(OUTPUT_DIR + '/dataset/picdata', train=True, transform=transforms, download=False)
# testData = cifar.CIFAR10(OUTPUT_DIR + '/dataset/picdata', train=False, transform=transforms)
# train_loader = DataLoader(trainData, batch_size=64, shuffle=False)
# testLoader = DataLoader(testData, batch_size=128, shuffle=False)
# # # 数据集可视化
# # examples = enumerate(trainLoader)
# # batchIndex, (imgData, labels) = next(examples)
# # fig = plt.figure()
# # for i in range(9):
# #     plt.subplot(3, 3, i + 1)
# #     plt.tight_layout()
# #     plt.imshow(imgData[i][0], interpolation='none')
# #     plt.title("{}".format(classes[labels[i]]))
# #     plt.xticks([])
# #     plt.yticks([])
# # plt.show()
# a = torchvision.models.resnet18()
# Model = common_config_model
# max_epoch = 40
# epoch_set = [1, 5, 10, 15, 20, 30, 40, 80, 120, 160, 200, 300, 400, 500, 600, 800, 1000]
# lr = common_config_learning_rate
# net = util.get_net(Model)  # Training Model
# # net = torchvision.models.resnet50()
# net = net.cuda()
# epochs_name = str(max_epoch)
# batch_size = common_config_batch_size
#
# print('epochs:', max_epoch, 'lr:', lr)
# print('net create success:', net)
# device_count = common_config_device_count
#
# train_data_set_type = common_config_train_set
# test_data_set_type = common_config_test_set
# P = common_config_train_point
# testP = common_config_test_point
# snr = common_config_train_snr
# test_snr = common_config_test_snr
# snr_name_train = util.buildSNR(snr)
# snr_name_test = util.buildSNR(test_snr)
# c_name = util.buildC(device_count)
# diff_channel = common_config_diff_channel
#
#
# def save_weights(epoch, filename):
#     """Save netG and netD weights for the current epoch.
#     Args:
#         :param epoch:
#         :param filename:
#     """
#     # weight_dir = os.path.join('./output/transformModel', 'train', 'weights')
#     # weight_dir = os.path.join('./output')
#     weight_dir = OUTPUT_DIR + '/model/' + train_data_set_type
#     if not os.path.exists(weight_dir):
#         os.makedirs(weight_dir)
#     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict()},
#                '%s/%s' % (weight_dir, filename))
#
#
# # 第三步：训练网络
# def train(para):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#     # model_file = 'output/M1_NormalAug_TR-P-1-2-3-4_C10_S-no_EP500_LR0-0001.pth'
#     # net.load_state_dict(torch.load(model_file))
#     net.train()
#     x = list()
#     loss_list = list()
#     for epoch in range(1, max_epoch + 1):
#         running_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         best_acc = 0
#         acc = 0
#         epoch_loss = 0
#         for i, data in enumerate(train_loader, 0):
#             inputs, train_labels = data
#             inputs, labels = Variable(inputs), Variable(train_labels)
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#
#             optimizer.zero_grad()  # 将梯度初始化为零
#             outputs = net(inputs)
#             _, train_predicted = torch.max(outputs.detach(), 1)
#             # 第二个返回值是找到最大值的索引位置(此处是0,1)
#             train_correct += (train_predicted == labels.data).sum()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             # 测试性能
#             train_total += train_labels.size(0)
#             epoch_loss = running_loss / train_total
#             acc = 100 * train_correct / train_total
#             print('[%d] 执行次数 %d' % (epoch, i))
#         if epoch in epoch_set and acc > best_acc:
#             best_acc = acc
#             save_weights(epoch, build_para_name(str(epoch)))
#         # print_current_performance(acc, best_acc)
#         x.append(epoch)
#         loss_list.append(epoch_loss)
#         print('train %d epoch loss: %.6f  acc: %.6f   best_acc:  %.6f' %
#               (epoch, epoch_loss, acc, best_acc))
#     plt.figure('Loss')
#     plt.plot(x, loss_list, "r-")
#     plt.show()
#
#
# def build_para_name(epochs_name):
#     return Model + '_' + train_common + '_EP' + epochs_name + '_LR' + util.buildLR(lr) + '.pth'
#
#
# if __name__ == "__main__":
#     # Config
#     # train_common = util.buildP(P, 'TR-All') + '_' + c_name + '_' + snr_name_train
#     # para = Model + '_' + train_common + '_ep' + str(epochs) + '.pth'
#     train_common = train_data_set_type + '_' + util.buildP(P, 'TR') \
#                    + '_' + c_name + '_' + snr_name_train
#     # 训练后的模型参数命名
#     para = build_para_name(epochs_name)
#     txt_dir = OUTPUT_DIR + '/class' + '/' + train_data_set_type + '/'
#     train_set_file = txt_dir + train_common + '.txt'
#     test_set_file = txt_dir + test_data_set_type + '_' \
#                     + util.buildP(testP, 'TE') + '_' + c_name + '_' \
#                     + snr_name_test + '.txt'
#     cm_name = Model + '_' + c_name + '_' + util.buildP(P, 'TR') + '_TR' + util.buildSNR(snr) \
#               + '_' + util.buildP(testP, 'TE') + '_TE' + util.buildSNR(test_snr) \
#               + '_EP' + epochs_name \
#               + '_LR' + util.buildLR(lr)
#     print('Model name is', para)
#     print('Train filename', train_set_file)
#     print('Test filename', test_set_file)
#     print('CM Name', cm_name)
#
#     # 第一步：数据加载与处理,选择训练设备cpu/gpu
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     print("是否使用GPU加速：", torch.cuda.is_available())
#     # 为当前GPU/CPU设置随机种子(第三行是否必要存疑)
#     # torch.manual_seed(2020)
#     # torch.cuda.manual_seed(2020)
#     # np.random.seed(2020)
#     # train_data = MultiChannelDataset(label_file=train_set_file, type='train', device_count=device_count, diff_channel=diff_channel,
#     #                        transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]))
#     # train_data = MyDataset(txt=train_set_file, type='train', device_count=device_count,
#     #                        transform=transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()]))
#     # 此处的shuffle只是在一个小批次内进行打乱
#     # train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
#     # gpu版本的话，num_workers可以提高到8，此处的batchsize有意义么？只是加载数据多少而已？
#     # test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
#     print('create data_loader successfully!')
#     print('parameters:', '\ndata_type:', train_common, '\nbatch_size:', batch_size)
#     train(para)
