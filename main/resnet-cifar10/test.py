import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import matplotlib.pyplot as plt
from dataset import WiFiDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_set_file = '/mnt/DiskA-1.7T/ztz/Output/DCTF_RFF/class/ClearedDataset-20_ModelB_DS_S401-500/' \
                'ClearedDataset-20_ModelB_DS_S401-500_TE-P-1-2_C9_S-no.txt'
test_data = WiFiDataset(txt=test_set_file, type='test', device_count=9, shuffle=True,
                        transform=transforms.Compose([transforms.CenterCrop(64),
                                                      transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=2,
                                          shuffle=False, pin_memory=True)
net = ResNet18().to(device)
net.load_state_dict(torch.load('/mnt/DiskA-1.7T/ztz/Output/ResNet-18-cifar10/model/net_020.pth'))
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
