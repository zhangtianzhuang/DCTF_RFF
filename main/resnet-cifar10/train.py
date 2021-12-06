import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import matplotlib.pyplot as plt
from dataset import WiFiDataset

dataset_dir = '/mnt/DiskA-1.7T/ztz/CommonDataset/cifar-10'
output_dir = '/mnt/DiskA-1.7T/ztz/Output/ResNet-18-cifar10/'

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# 输出结果保存路径
parser.add_argument('--outf', default=output_dir + 'model/', help='folder to output images and model checkpoints')
# 恢复训练时的模型路径
parser.add_argument('--net', default=output_dir + 'model/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()

switch_flag = [False, False]
switch_point = [135, 185, 240]

# 超参数设置
EPOCH = switch_point[2]  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.1  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 训练数据集
# trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=False, transform=transform_train)
# # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# # Cifar-10的标签
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# example_1 = enumerate(trainloader)
# _, (imgData, labels) = next(example_1)
# print('imgData_1:', imgData.shape)
# print('lable_1:', labels)


train_set_file = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/ClearedDataset-20_ModelB_DS_S1-400/' \
                 'ClearedDataset-20_ModelB_DS_S1-400_TR-P-1-2_C9_S-5-10-15-20-25-30-no.txt'
test_set_file = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/ClearedDataset-20_ModelB_DS_S401-500/' \
                'ClearedDataset-20_ModelB_DS_S401-500_TE-P-1-2_C9_S-10-15-20-25-30-no.txt'
train_data = WiFiDataset(txt=train_set_file, type='train', device_count=9,
                         transform=transforms.Compose([transforms.CenterCrop(64),
                                                       transforms.ToTensor()]))
# 此处的shuffle只是在一个小批次内进行打乱
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, num_workers=2, shuffle=True)

test_data = WiFiDataset(txt=test_set_file, type='test', device_count=9, shuffle=True,
                        transform=transforms.Compose([transforms.CenterCrop(64),
                                                      transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=2,
                                          shuffle=False, pin_memory=True)
# example_2 = enumerate(train_loader)
# _, (imgData2, labels2) = next(example_2)
# print('imgData_2:', imgData2.shape)
# print('lable_2:', labels2)

# 模型定义-ResNet
net = ResNet18().to(device)

torchvision.models.resnet18()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer_2 = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
                        weight_decay=5e-4)
optimizer_3 = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                        weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            train_x = list()
            train_y = list()
            test_x = list()
            test_y = list()
            for epoch in range(pre_epoch, EPOCH):
                if epoch >= switch_point[0] and not switch_flag[0]:
                    print('>>切换到LR=0.01')
                    optimizer = optimizer_2
                    switch_flag[0] = True
                if epoch >= switch_point[1] and not switch_flag[1]:
                    print('>>切换到LR=0.001')
                    optimizer = optimizer_3
                    switch_flag[1] = True
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    train_x.append(i + 1 + epoch * length)
                    train_y.append(100. * correct / total)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    test_x.append(epoch + 1)
                    test_y.append(acc)
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d, Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
            print("train_x:", train_x)
            print("train_y:", train_y)
            print("test_x:", test_x)
            print("test_y:", test_y)
            plt.figure('error rate')
            plt.plot(train_x, train_y, 'r-')
            plt.plot(test_x, test_y, 'b-')
            plt.show()
